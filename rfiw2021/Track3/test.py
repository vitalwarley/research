import os
import sys

# add dir parent to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import gc
import sys

import torch
from sklearn.metrics import auc, roc_curve
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from Track3.dataset import *
from Track3.models_track1 import *
from Track3.utils import *


def sort_list(l):
    l.sort(key=lambda n: n[1], reverse=True)
    return [int(now[0]) for now in l]


def write_csv(line, path):
    for i in range(len(line)):
        line[i] = str(line[i])
    with open(path, "a+", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(line)


@torch.no_grad()
def test(args):
    probe_path = os.path.join(args.sample_root, "probe.txt")
    gallery_path = os.path.join(args.sample_root, "gallery.txt")

    gallery_batch_size = args.batch_size
    model_path = args.model_path
    log_path = args.log_path
    submit_path = args.pred_path

    # make dirs: log_path and submit_path
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.exists(os.path.dirname(submit_path)):
        os.makedirs(os.path.dirname(submit_path))

    probe = Probe(probe_path)
    gallery = Gallery(gallery_path)
    gallery_loader = DataLoader(gallery, batch_size=gallery_batch_size, num_workers=1, pin_memory=False)
    model = NetTrack3(model_path).cuda()
    model.eval()

    # make pbar from tqdm for prob * gallery
    pbar = tqdm(total=len(probe) * len(gallery_loader), desc="Predicting...")

    for i in range(len(probe)):
        probe_index, probe_images = probe[i]
        probe_embedings = model(probe_images)
        probe_res_list = []
        for gallery_index, gallery_images in gallery_loader:
            gallery_embedings = model(gallery_images)
            if args.score == "max":
                score = torch.max(
                    torch.cosine_similarity(
                        torch.unsqueeze(gallery_embedings, dim=1), torch.unsqueeze(probe_embedings, dim=0), dim=2
                    ),
                    dim=1,
                )[0]
            else:
                score = torch.mean(
                    torch.cosine_similarity(
                        torch.unsqueeze(gallery_embedings, dim=1), torch.unsqueeze(probe_embedings, dim=0), dim=2
                    ),
                    dim=1,
                )

            tmp_res = torch.cat([torch.unsqueeze(gallery_index, dim=1), torch.unsqueeze(score, dim=1)], dim=1)
            probe_res_list.extend(tmp_res.cpu().detach().numpy().tolist())
            pbar.update(1)
        probe_res_list = sort_list(probe_res_list)
        write_csv(probe_res_list, submit_path)
        # mylog(probe_index," is over ,","len is " , len(probe_res_list),path=log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create predation task3")
    parser.add_argument("--sample_root", default="./sample", type=str, help="test file root default ./sample")
    parser.add_argument("--model_path", type=str, help="model save path")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size default 20")
    parser.add_argument("--score", type=str, default="mean", help="mean or max")
    parser.add_argument("--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument("--pred_path", default="./predictions.csv", type=str, help="predicted file save location")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    test(args)
