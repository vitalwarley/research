import argparse
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import mxnet as mx
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import torch
import torchmetrics as tm
from matplotlib import pyplot as plt
from more_itertools import grouper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mytypes as t
from dataset import FamiliesDataset, ImgDataset, PairDataset
from model import Model
from tasks import init_fiw, init_ms1m
from utils import load_lfw, load_pairs, log_results


def predict(model, datamodule, is_lfw: bool):
    distances = []
    similarities = []
    labels = []
    for batch_idx, (first, second, label) in enumerate(
        tqdm(datamodule.val_dataloader())
    ):
        if is_lfw:
            # unflipped images
            first = first[0]
            second = second[0]
        distance, similarity = model(first, second, label, batch_idx)
        distances.append(distance)
        similarities.append(similarity)
        labels.append(label.numpy())
    distances = np.concatenate(distances)
    similarities = np.concatenate(similarities)
    labels = np.concatenate(labels)
    return distances, similarities, labels


class CompareModel(object):
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.writer = None

    def _load_model(self):
        raise NotImplementedError

    def get_embedding(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def metric(self, emb1, emb2):
        _sum = np.sum(emb1 * emb2, axis=-1)
        _norms = np.linalg.norm(emb1, axis=-1) * np.linalg.norm(emb2, axis=-1)
        return _sum / _norms

    def _log_embeddings(self, images, labels, embeddings, batch_idx):
        bs = len(labels)
        pair_names = np.array(["IMG1", "IMG2"]).repeat(bs)
        sample_idx_str_arr = np.tile([f"_SAMPLE_{i}" for i in range(bs)], 2)
        label_str_arr = np.tile([f"_LABEL_{i.item()}" for i in labels], 2)
        labels_str = np.char.add(
            np.char.add(pair_names, sample_idx_str_arr), label_str_arr
        )
        model_name = self.__class__.__name__
        writer.add_embedding(
            embeddings,
            metadata=labels_str,
            label_img=images,
            global_step=batch_idx,
            tag=f"{model_name}/embeddings",
        )

        positive_samples = labels == 1
        negative_samples = labels == 0

        if any(positive_samples):
            writer.add_histogram(
                f"{model_name}/embeddings distribution/postive samples/image 1",
                embeddings[:bs][labels == 1],
                global_step=batch_idx,
            )
            writer.add_histogram(
                f"{model_name}/embeddings distribution/positive samples/image 2",
                embeddings[bs:][labels == 1],
                global_step=batch_idx,
            )

        if any(negative_samples):
            writer.add_histogram(
                f"{model_name}/embeddings distribution/negative samples/image 1",
                embeddings[:bs][labels == 0],
                global_step=batch_idx,
            )
            writer.add_histogram(
                f"{model_name}/embeddings distribution/negative samples/image 2",
                embeddings[bs:][labels == 0],
                global_step=batch_idx,
            )

    def __call__(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        label: torch.Tensor,
        batch_idx: int,
    ) -> float:
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        sim = self.metric(emb1, emb2)
        normed_emb1 = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
        normed_emb2 = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)
        diff = normed_emb1 - normed_emb2
        dist = np.linalg.norm(diff, axis=-1)

        if batch_idx % 5 == 0:
            embs = np.concatenate([emb1, emb2])
            imgs = torch.cat([img1, img2])
            self._log_embeddings(imgs, label.numpy(), embs, batch_idx)

        return dist, sim


class CompareModelTorch(CompareModel):
    def __init__(self, insightface: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model = Model(weights=self.model_name, insightface=insightface)
        self.model.eval()
        self.model.to(torch.device(self.device))

    def get_embedding(self, img: torch.Tensor) -> t.Embedding:
        embs, _ = self.model(img.to(torch.device(self.device)))
        return embs.detach().cpu().numpy()


class CompareModelMXNet(CompareModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx = mx.cpu() if self.device == "cpu" else mx.gpu()
        self._load_model()

    def _load_model(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_name, 0)
        sym = sym.get_internals()["fc1_output"]
        model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        data_shape = (1, 3, 112, 112)
        model.bind(data_shapes=[("data", data_shape)], for_training=False)
        model.set_params(arg_params, aux_params)
        # warmup
        data = mx.nd.zeros(shape=data_shape)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        _ = model.get_outputs()[0].asnumpy()
        self.model = model

    def get_embedding(self, img) -> t.Embedding:
        img = img.cpu().numpy()
        # revert transform because baseline model was trained on np.uint8 images
        img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        img = mx.nd.array(img)
        batch = mx.io.DataBatch([img])
        self.model.forward(batch, is_train=False)
        out = self.model.get_outputs()[0].asnumpy()
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-models", type=str, default="both", choices=["both", "torch", "mxnet"]
    )
    parser.add_argument("--torch-model", type=str)
    parser.add_argument("--mxnet-model", type=str)
    parser.add_argument("--insightface", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # add time.time() to outputs path
    time_str = str(int(time.time()))
    output_dir = Path("outputs", args.dataset, time_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(str(output_dir))

    # TODO: refactor inference for each model in separated functions

    if args.dataset == "fiw":
        datamodule = init_fiw(
            data_dir="/home/warley/dev/datasets/fiw",
            batch_size=args.batch_size,
            mining_strategy="baseline",
            num_workers=8,
        )
    elif args.dataset == "lfw":
        datamodule = init_ms1m(
            data_dir="/home/warley/dev/datasets/MS1M_v3", batch_size=128
        )
    datamodule.setup("validate")

    # TODO: accumulate results and log all at once?

    if args.run_models in ["both", "torch"]:
        ### TORCH
        model = CompareModelTorch(
            model_name=args.torch_model,
            device=args.device,
            insightface=args.insightface,
        )
        model.writer = writer  # skip save_hyperparams error
        # TODO: add arg to use both images if lfw?
        distances, similarities, y_true = predict(
            model, datamodule, is_lfw=(args.dataset == "lfw")
        )
        best_threshold, best_accuracy, auc_score = log_results(
            writer, "torch", distances, similarities, y_true
        )
        print("Torch results:")
        print(f"\tbest_threshold: {best_threshold}")
        print(f"\tbest_accuracy: {best_accuracy}")
        print(f"\tauc_score: {auc_score}")

        del model
        torch.cuda.empty_cache()

    if args.run_models in ["both", "mxnet"]:
        # MXNET
        model = CompareModelMXNet(model_name=args.mxnet_model, device=args.device)
        model.writer = writer
        distances, similarities, y_true = predict(
            model, datamodule, is_lfw=(args.dataset == "lfw")
        )
        best_threshold, best_accuracy, auc_score = log_results(
            writer, "mxnet", distances, similarities, y_true
        )
        print("MXNET results:")
        print(f"\tbest_threshold: {best_threshold}")
        print(f"\tbest_accuracy: {best_accuracy}")
        print(f"\tauc_score: {auc_score}")

    writer.close()
