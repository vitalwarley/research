# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Aim: predict age and gender from RFIW2021 data -- train and val sets.
#
# 1. Inspect MiVOLO to understand its setup
# 2. Copy the necessary code
# 3. Adapt it to save each prediction to a dataframe (maybe extending val.txt to have age and gender)
# 4. Analyse the generated dataframe

# %% [markdown]
# ## Import libraries

# %%
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from mivolo.data.data_reader import get_all_files
from mivolo.predictor import Predictor
from tqdm import tqdm

# %% [markdown]
# # Set up paths

# %%
# %%
try:
    IS_NOTEBOOK = True
    HERE = Path(__file__).resolve().parent
except NameError:
    IS_NOTEBOOK = False
    HERE = Path().resolve()

DATA_DIR = Path(HERE, "../rfiw2021/Track1/Validation/val-faces")
MODELS_DIR = Path(HERE, "../MiVOLO/models")


# %%
@dataclass
class Config:
    device: str = "cuda"
    detector_weights: str = MODELS_DIR / "yolov8x_person_face.pt"
    checkpoint: str = MODELS_DIR / "mivolo_imbd.pth.tar"
    with_persons: bool = False
    disable_faces: bool = False
    draw = False


# %%
args = Config()
predictor = Predictor(args)

# %%
image_files = get_all_files(DATA_DIR)

preds = []

for img_p in tqdm(image_files[:10]):
    img = cv2.imread(img_p)
    pred, out_im = predictor.recognize(img)
    img_info = {k: v for k, v in zip(["fid", "mid", "pid_filename"], Path(img_p).parts[-3:])}
    pred_info = {"age": pred.ages[1], "gender": pred.genders[1], "gender_score": pred.gender_scores[1]}
    preds.append({**img_info, **pred_info})

# %%
preds

# %%
df = pd.DataFrame(preds)

# %%
df

# %%
