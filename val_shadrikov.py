import time
import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np
import mxnet as mx
import onnxruntime as ort
import pandas as pd
import torch
import torchmetrics as tm
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from more_itertools import grouper

import mytypes as t
from dataset import FamiliesDataset, PairDataset, ImgDataset
from model import Model
from tasks import init_ms1m, init_fiw
from utils import load_pairs, load_lfw, make_accuracy_vs_threshold_plot, make_roc_plot


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


def predict_on_datamodule(model, datamodule, is_lfw: bool):
    distances = []
    similarities = []
    labels = []
    for first, second, label in tqdm(datamodule.val_dataloader()):
        if is_lfw:
            # unflipped images
            first = first[0]
            second = second[0]
        distance, similarity = model(first, second)
        distances.append(distance)
        similarities.append(similarity)
        labels.append(label.numpy())
    distances = np.concatenate(distances)
    similarities = np.concatenate(similarities)
    labels = np.concatenate(labels)
    return distances, similarities, labels


def predict_on_lfw(model):
    similarities = []
    labels = []
    for (first, second), label in tqdm(
        load_lfw("/home/warley/dev/datasets/MS1M_v3", "lfw.bin", (112, 112)), total=6000
    ):
        similarity = model(first, second)
        similarities.append(similarity)
        labels.append(label)
    return np.stack(similarities, axis=0), np.stack(labels, axis=0)


def predict(
    model: Callable[[Path, Path], t.Labels], pair_list: List[t.PairPath]
) -> t.Labels:
    distances = []
    similarities = []
    for idx, (path1, path2) in tqdm(enumerate(pair_list), total=len(pair_list)):
        distance, similarity = model(path1, path2)
        distances.append(distance)
        similarities.append(similarity)
    distances = np.concatenate(distances)
    similarities = np.concatenate(similarities)
    return distances, similarities


class CompareModel(object):
    def __init__(self, model_name: str, transform: bool, device: str):
        self.model_name = model_name
        self.transform = transform
        self.device = device
        self.embeddings = {"first": [], "second": []}

    def _load_model(self):
        raise NotImplementedError

    def get_embedding(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def metric(self, emb1, emb2):
        _sum = np.sum(emb1 * emb2, axis=-1)
        _norms = np.linalg.norm(emb1, axis=-1) * np.linalg.norm(emb2, axis=-1)
        return _sum / _norms

    def __call__(self, path1: Path, path2: Path) -> float:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        self.embeddings["first"].append(emb1)
        self.embeddings["second"].append(emb2)
        sim = self.metric(emb1, emb2)
        normed_emb1 = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
        normed_emb2 = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)
        diff = normed_emb1 - normed_emb2
        dist = np.linalg.norm(diff, axis=-1)
        return dist, sim


class CompareModelTorch(CompareModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Model(weights=self.model_name)
        self.model.eval()
        self.model.to(torch.device(self.device))

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if isinstance(im_path, Path):
            img = cv2.imread(str(im_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = im_path
        if self.transform:
            # raw data
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = np.expand_dims(img, 0)
            # acc = 0.6 with it, but 0.5 without it
            # because my model was training with images scaled in this way
            img = ((img / 255.0) - 0.5) / 0.5
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).to(torch.device(self.device))
            embs, _ = self.model(img)
        else:
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

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if isinstance(im_path, Path):
            img = mx.img.imread(str(im_path))
        else:
            img = im_path
        if self.transform:
            # it seems that the model was trained with images on scale (0, 255)
            img = (
                mx.nd.array(img).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
            )
        else:  # for datamodule samples
            img = img.cpu().numpy()
            # acc = 0.88 with it, but 0.5 without it
            # if img is not normalized (img - 0.5) / 0.5
            # then acc goes to 0.987 (same as with original scheme)
            img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(
                np.uint8
            )  # in datamodule i scale, therefore here i scale back
            img = mx.nd.array(img)
        batch = mx.io.DataBatch([img])
        self.model.forward(batch, is_train=False)
        out = self.model.get_outputs()[0].asnumpy()
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--datamodule", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # add time.time() to outputs path
    time_str = str(int(time.time()))
    output_dir = Path(
        "outputs", args.dataset, "datamodule" if args.datamodule else "raw", time_str
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(str(output_dir))

    # TODO: refactor inference for each model in separated functions

    if args.dataset == "fiw":
        pair_list, y_true = load_pairs()
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

    # TODO: accumulate results and log all at once?

    ### TORCH
    model = CompareModelTorch(
        model_name="../training/research/lightning_logs/version_24/checkpoints/epoch=34-step=49770.ckpt",
        transform=not args.datamodule,
        device=args.device,
    )
    if args.datamodule:
        # TODO: add arg to use both images if lfw
        datamodule.setup("validate")
        distances, similarities, y_true = predict_on_datamodule(
            model, datamodule, is_lfw=(args.dataset == "lfw")
        )
        log_results(writer, "torch", distances, similarities, y_true)
    else:
        # TODO: fix for lfw
        distances, similarities = predict(model, pair_list)
        log_results(writer, "torch", distances, similarities, y_true)

    del model
    torch.cuda.empty_cache()

    # MXNET
    model = CompareModelMXNet(
        model_name="arcface_r100_v1", transform=not args.datamodule, device=args.device
    )
    if args.datamodule:
        distances, similarities, y_true = predict_on_datamodule(
            model, datamodule, is_lfw=(args.dataset == "lfw")
        )
        log_results(writer, "mxnet", distances, similarities, y_true)
    else:
        # TODO: fix for lfw
        distances, similarities = predict(model, pair_list)
        log_results(writer, "mxnet", distances, similarities, y_true)

    writer.close()
