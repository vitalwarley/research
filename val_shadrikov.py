import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np
import mxnet as mx
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from matplotlib import pyplot as plt
from more_itertools import grouper

import mytypes as t
from dataset import FamiliesDataset, PairDataset, ImgDataset
from model import PretrainModel
from tasks import init_ms1m


def load_lfw(root: str, target: str, _image_size):
    bin_path = Path(root) / target
    # read bin
    with open(bin_path, "rb") as f:
        bins, labels = pickle.load(f, encoding="bytes")
    for idx, (first, second) in enumerate(grouper(bins, 2)):
        if first is None or second is None:
            continue
        first = cv2.imdecode(first, cv2.IMREAD_COLOR)
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        first = cv2.resize(first, _image_size)
        second = cv2.imdecode(second, cv2.IMREAD_COLOR)
        second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
        second = cv2.resize(second, _image_size)
        # in EvalPretrainDataset I return the scaled image,
        # but here I return the original image.
        # For my onnx, I perform scaling on get_embeddings
        # for the mxnet, we don't need scaling.
        yield (first, second), labels[idx]


def predict_on_lfw_datamodule(model, datamodule):
    similarities = []
    labels = []
    for first, second, label in tqdm(datamodule.val_dataloader()):
        first = first[0]
        second = second[0]
        similarity = model(first, second)
        similarities.append(similarity)
        labels.append(label.numpy())
    return np.concatenate(similarities), np.concatenate(labels)


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
    predictions = []
    for idx, (path1, path2) in tqdm(enumerate(pair_list), total=len(pair_list)):
        cur_prediction = model(path1, path2)
        predictions.append(cur_prediction)
    return np.stack(predictions, axis=0)


class CompareModel(object):
    def __init__(self, model_name: str, transform: bool = True):
        self.model_name = model_name
        self.transform = transform

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
        return self.metric(emb1, emb2)


class CompareModelONNX(CompareModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._load_model()

    def _load_model(self):
        self.session = ort.InferenceSession(self.model_name, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if isinstance(im_path, Path):
            img = cv2.imread(str(im_path))
        else:
            img = im_path
        if self.transform:
            # raw data
            img = img.transpose(2, 0, 1).reshape(1, 3, 112, 112).astype(np.float32)
            # acc = 0.6 with it, but 0.5 without it
            img = ((img / 255.0) - 0.5) / 0.5
        else:
            # datamodule
            img = img.cpu().numpy()
        return self.session.run([self.output_name], {self.input_name: img})[0]


class CompareModelMXNet(CompareModel):
    def __init__(self, ctx: mx.Context = mx.cpu(), **kwargs):
        super().__init__(**kwargs)
        root = Path("fitw2020") / "models"
        self.model_name = str(root / self.model_name)
        self.ctx = ctx
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
            img = (
                mx.img.imread(str(im_path))
                .transpose((2, 0, 1))
                .expand_dims(0)
                .astype(np.float32)
            )
        else:
            if self.transform:
                img = (
                    mx.nd.array(im_path)
                    .transpose((2, 0, 1))
                    .expand_dims(0)
                    .astype(np.float32)
                )
            else:
                img = im_path
                img = img.cpu().numpy()
                # acc = 0.88 with it, but 0.5 without it
                # if img is not normalized (img - 0.5) / 0.5
                # then acc goes to 0.987 (same as with original scheme)
                img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
                img = mx.nd.array(img)
        batch = mx.io.DataBatch([img])
        self.model.forward(batch, is_train=False)
        out = self.model.get_outputs()[0].asnumpy()
        return out


def load_pairs():
    pairs = []
    root_path = Path("/home/warley/dev/datasets/fiw/val-faces-det")
    with open("/home/warley/dev/datasets/fiw/val_pairs.csv", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue
            img1, img2, label = line.split(",")
            pairs.append((root_path / img1, root_path / img2, int(label)))
    y_true = [label for _, _, label in pairs]
    pair_list = [(img1, img2) for img1, img2, _ in pairs]
    return pair_list, y_true


if __name__ == "__main__":
    pair_list, y_true = load_pairs()
    datamodule = init_ms1m(data_dir="/home/warley/dev/datasets/MS1M_v3")
    datamodule.setup("validate")

    model = CompareModelONNX(model_name="my_pretrained_model.onnx")
    predictions, y_true = predict_on_lfw(model)
    y_pred = predictions > 0.5
    print(f"Accuracy (torch as onnx) on raw data: {accuracy_score(y_true, y_pred)}")
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, "-r", lw=1, label=f"lfw AUC (my onnx model):{auc(fpr, tpr):.4f}")

    model = CompareModelONNX(model_name="my_pretrained_model.onnx", transform=False)
    predictions, y_true = predict_on_lfw_datamodule(model, datamodule)
    y_pred = predictions > 0.5
    print(f"Accuracy (torch as onxx) on datamodule: {accuracy_score(y_true, y_pred)}")
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(
        fpr,
        tpr,
        "--r",
        lw=1,
        label=f"lfw datamodule AUC (my onnx model):{auc(fpr, tpr):.4f}",
    )

    model = CompareModelMXNet(model_name="arcface_r100_v1", ctx=mx.gpu())
    predictions, y_true = predict_on_lfw(model)
    y_pred = predictions > 0.5
    print(f"Accuracy (mxnet) on raw data: {accuracy_score(y_true, y_pred)}")
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(
        fpr,
        tpr,
        "b",
        lw=1,
        label=f"lfw AUC (insightface mxnet):{auc(fpr, tpr):.4f}",
    )

    model = CompareModelMXNet(
        model_name="arcface_r100_v1", ctx=mx.gpu(), transform=False
    )
    predictions, y_true = predict_on_lfw_datamodule(model, datamodule)
    y_pred = predictions > 0.5
    print(f"Accuracy (mxnet) on datamodule: {accuracy_score(y_true, y_pred)}")
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(
        fpr,
        tpr,
        "--b",
        lw=1,
        label=f"lfw datamodule AUC (insightface mxnet):{auc(fpr, tpr):.4f}",
    )

    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc.pdf", transparent=True, pad_inches=0, bbox_inches="tight")
    plt.show()
