import gc
import os
import sys

from sklearn.metrics import accuracy_score

FILE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{FILE}/..")

import argparse

import torch
from torch import nn
from torch.optim import SGD
from torchmetrics.functional import accuracy
from tqdm import tqdm
from Track1.dataset import *
from Track1.losses import *
from Track1.models import *


def training(args):
    batch_size = args.batch_size
    val_batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = 50
    save_path = args.save_path
    beta = args.beta
    log_path = args.log_path

    train_dataset = FIW(os.path.join(args.sample, "train_sort.txt"), classification=True)
    val_dataset = FIW(os.path.join(args.sample, "val_choose.txt"), classification=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=False, shuffle=True, generator=torch.Generator('cuda'))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=8, pin_memory=False)

    model = NetClassifier().cuda()

    optimizer_model = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    max_acc = 0.0

    ce_loss = nn.CrossEntropyLoss()

    for epoch_i in tqdm(range(epochs)):
        mylog("\n*************", path=log_path)
        mylog("epoch " + str(epoch_i + 1), path=log_path)
        ce_loss_epoch = 0

        model.train()

        for index_i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image1, image2, labels = data

            # e1,e2,x1,x2= model([image1,image2])
            # loss = ce_loss(x1,x2,beta=beta)

            preds = model([image1, image2])
            loss = ce_loss(preds, labels)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            ce_loss_epoch += loss.item()

            # if (index_i + 1) == steps_per_epoch:
            #     break

        # use_sample = (epoch_i + 1) * batch_size * steps_per_epoch
        # train_dataset.set_bias(use_sample)

        mylog("ce_loss:" + "%.6f" % (ce_loss_epoch / index_i), path=log_path)
        model.eval()
        with torch.no_grad():
            acc = val_model(model, val_loader)
        mylog("acc is %.6f " % acc, path=log_path)
        if max_acc < acc:
            mylog("acc improve from :" + "%.6f" % max_acc + " to %.6f" % acc, path=log_path)
            max_acc = acc
            mylog("save model " + save_path, path=log_path)
            save_model(model, save_path)
        else:
            mylog("acc did not improve from %.6f" % float(max_acc), path=log_path)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def val_model(model, val_loader):
    y_true = []
    y_pred = []
    for img1, img2, labels in val_loader:
        preds = model([img1.cuda(), img2.cuda()])
        y_pred.extend(preds)
        y_true.extend(labels)
    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    acc = accuracy(y_pred, y_true, task="multiclass", num_classes=11)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=25, help="batch size default 25")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--epochs", type=int, default=80, help="epochs number default 80")
    parser.add_argument("--beta", default=0.08, type=float, help="beta default 0.08")
    parser.add_argument("--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)
