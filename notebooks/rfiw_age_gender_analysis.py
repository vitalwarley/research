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
errors = []

for img_p in tqdm(image_files):
    img = cv2.imread(img_p)
    try:
        pred, out_im = predictor.recognize(img)
        pred_info = {"age": pred.ages[1], "gender": pred.genders[1], "gender_score": pred.gender_scores[1]}
    except:
        errors.append(img_p)
        pred_info = {"age": 0, "gender": "U", "gender_score": 0}
    img_info = {k: v for k, v in zip(["fid", "mid", "pid_filename"], Path(img_p).parts[-3:])}
    preds.append({**img_info, **pred_info})

print(f"Errors found: {len(errors)}")

# %%
errors[:10]

# %%
preds[:5]

# %%
ag_preds = pd.DataFrame(preds)

# %%
ag_preds

# %%
ag_preds.fillna(0, inplace=True)
ag_preds.replace({None: "unknown"}, inplace=True)
ag_preds


# %% [markdown]
# # Data Analysis
#
# - Age histogram for each kinship relation, for each face image
# - Gender histogram for each kinship relation, for each face index
#
# We are supposing that there are no intersections between faces in the pairs. That is, first faces aren't present as second faces in any other pair.
#
# What I found while coding:
#
# 1. Face pairs with repeated faces are possible for face 1.
#
# ## Steps
#
# 1. Load validation pairs info into a dataframe: `val_pairs`
# 2. Extract FID, MID, filename to new columns
# 3. Merge age and gender predictions `ag_preds` with `val_pairs`
#    - New dataframe will have: `face1_path, face2_path, kin_relation, is_kin, face1_family_id (fid1), face2_family_id (fid2), pid1_age, pid1_gender, pid2_age, pid2_gender`
# 4. Plot histograms: a unique plot with 2 subplots, where row 1 has age and row 2 has gender for both faces histogram densitys overlaid


# %%
def load_data(path: str):
    df = pd.read_csv(
        path,
        delimiter=" ",
        header=None,
        names=["id", "face1_path", "face2_path", "kin_relation", "is_kin"],
    )
    df.drop(["id"], inplace=True, axis=1)
    return df


val_pairs_fp = Path(HERE, "../rfiw2021/Track1/sample0/val.txt")
val_pairs_fp_model_sel = Path(HERE, "../rfiw2021/Track1/sample0/val_choose.txt")
val_pairs1 = load_data(val_pairs_fp)
val_pairs2 = load_data(val_pairs_fp_model_sel)
val_pairs = pd.concat([val_pairs1, val_pairs2]).reset_index(drop=True)
val_pairs

# %%
# Processing face1_path
face1_parts = val_pairs.face1_path.apply(lambda x: Path(x).parts[-3:])
val_pairs[["f1fid", "f1mid", "f1fp"]] = pd.DataFrame(face1_parts.tolist())

# Processing face2_path
face2_parts = val_pairs.face2_path.apply(lambda x: Path(x).parts[-3:])
val_pairs[["f2fid", "f2mid", "f2fp"]] = pd.DataFrame(face2_parts.tolist())
val_pairs

# %%
ag_preds

# %%
val_pairs = val_pairs.merge(
    ag_preds, left_on=["f1fid", "f1mid", "f1fp"], right_on=["fid", "mid", "pid_filename"], how="inner"
)
val_pairs.rename(columns={"age": "f1_age", "gender": "f1_gender", "gender_score": "f1_gender_score"}, inplace=True)
val_pairs.drop(["fid", "mid", "pid_filename"], axis=1, inplace=True)
val_pairs = val_pairs.merge(
    ag_preds, left_on=["f2fid", "f2mid", "f2fp"], right_on=["fid", "mid", "pid_filename"], how="inner"
)
val_pairs.rename(columns={"age": "f2_age", "gender": "f2_gender", "gender_score": "f2_gender_score"}, inplace=True)
val_pairs.drop(["fid", "mid", "pid_filename"], axis=1, inplace=True)
val_pairs

# %%
val_pairs.dtypes

# %% [markdown]
# ## Plot histograms

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# %%
import seaborn as sns

sns.set_style("whitegrid")

g = sns.FacetGrid(val_pairs, col="is_kin", row="kin_relation", height=2, aspect=3)
g = g.map(sns.kdeplot, "f1_age", color="red")
g = g.map(sns.kdeplot, "f2_age", color="blue")

# Create custom lines for the legend
red_line = mlines.Line2D([], [], color="red", label="Face 1")
blue_line = mlines.Line2D([], [], color="blue", label="Face 2")

# Adding the legend at the top
g.fig.legend(handles=[red_line, blue_line], loc="upper center", bbox_to_anchor=[0.5, 1.01], ncol=2)

plt.suptitle("Age by face order in the pair, by kinship", y=1.02)
plt.legend()
plt.show()
