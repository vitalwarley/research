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
from utils import load_lfw, load_pairs, make_accuracy_vs_threshold_plot, make_roc_plot


def log_results(writer, model_name, distances, similarities, y_true):
    writer.add_histogram(
        f"{model_name}/distances", distances[y_true == 0], global_step=0
    )
    writer.add_histogram(
        f"{model_name}/distances", distances[y_true == 1], global_step=1
    )
    writer.add_histogram(
        f"{model_name}/similarities", similarities[y_true == 0], global_step=0
    )
    writer.add_histogram(
        f"{model_name}/similarities", similarities[y_true == 1], global_step=1
    )

    fig = make_accuracy_vs_threshold_plot(
        distances, y_true, fn=lambda x, thresh: x < thresh
    )
    writer.add_figure(
        f"{model_name}/distances/accuracy vs threshold", fig, global_step=0
    )
    fig = make_accuracy_vs_threshold_plot(
        similarities, y_true, fn=lambda x, thresh: x > thresh
    )  # high sim, low dist
    writer.add_figure(
        f"{model_name}/similarities/accuracy vs threshold", fig, global_step=0
    )

    fig, auc_score = make_roc_plot(similarities, y_true)
    writer.add_scalar(f"roc_auc/{model_name}", auc_score, global_step=0)
    # writer.add_figure(f"{model_name}/roc", fig, global_step=0)

    similarities[similarities <= 0] = 0  # for plotting PR curve
    writer.add_pr_curve(
        f"{model_name}/similarities/pr curve", y_true, similarities, global_step=0
    )


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
        labels = np.char.add(np.char.add(pair_names, sample_idx_str_arr), label_str_arr)
        writer.add_embedding(
            embeddings,
            metadata=labels,
            label_img=images,
            global_step=batch_idx,
            tag=f"{self.__class__.__name__}/embeddings",
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Model(weights=self.model_name)
        self.model.eval()
        self.model.to(torch.device(self.device))

    def get_embedding(self, img: torch.Tensor) -> t.Embedding:
        embs, _ = self.model(img.to(torch.device(self.device)))
        return embs.detach().cpu().numpy()


class CompareModelMXNet(CompareModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = Path("fitw2020") / "models"
        self.model_name = str(root / self.model_name)
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
    parser.add_argument("--dataset", type=str)
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
            batch_size=128,
            mining_strategy="baseline",
            num_workers=8,
        )
    elif args.dataset == "lfw":
        datamodule = init_ms1m(
            data_dir="/home/warley/dev/datasets/MS1M_v3", batch_size=128
        )
    datamodule.setup("validate")

    # TODO: accumulate results and log all at once?

    ### TORCH
    model = CompareModelTorch(
        model_name="../training/research/lightning_logs/version_24/checkpoints/epoch=34-step=49770.ckpt",
        device=args.device,
    )
    model.writer = writer  # skip save_hyperparams error
    # TODO: add arg to use both images if lfw?
    distances, similarities, y_true = predict(
        model, datamodule, is_lfw=(args.dataset == "lfw")
    )
    log_results(writer, "torch", distances, similarities, y_true)

    del model
    torch.cuda.empty_cache()

    # MXNET
    model = CompareModelMXNet(model_name="arcface_r100_v1", device=args.device)
    model.writer = writer
    distances, similarities, y_true = predict(
        model, datamodule, is_lfw=(args.dataset == "lfw")
    )
    log_results(writer, "mxnet", distances, similarities, y_true)

    writer.close()
