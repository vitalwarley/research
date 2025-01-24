# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import math
import random
from itertools import combinations

import numpy as np

# %%
import pandas as pd

# %%
# Initialize variables
n_rows = 33
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
age_diff_distributions = {
    "fs": ("gaussian", 26.11, 11.24),
    "fd": ("gaussian", 27.88, 11.48),
    "ms": ("gaussian", 22.52, 9.88),
    "md": ("gaussian", 22.53, 10.02),
    "gfgs": ("gaussian", 38.24, 18.47),
    "gfgd": ("gaussian", 37.90, 18.31),
    "gmgs": ("gaussian", 32.78, 17.04),
    "gmgd": ("gaussian", 35.53, 17.63),
    "bb": ("power-law", 1, 1),
    "ss": ("power-law", 1, 1),
    "sibs": ("power-law", 1, 1),
}
gender_constraints = {
    "fs": ("male", "male"),
    "fd": ("male", "female"),
    "ms": ("female", "male"),
    "md": ("female", "female"),
    "gfgs": ("male", "male"),
    "gfgd": ("male", "female"),
    "gmgs": ("female", "male"),
    "gmgd": ("female", "female"),
    "bb": ("male", "male"),
    "ss": ("female", "female"),
    "sibs": ("either", "either"),
}


# %%
# Function to generate age difference
def generate_age_diff(distribution, *params):
    if distribution == "gaussian":
        mean, std_dev = params
        return max(0, np.random.normal(mean, std_dev))
    else:
        constant, exponent = params
        return max(0, np.random.pareto(exponent) + constant)


# %%
# Create df_final
df_final = pd.DataFrame(columns=["x1", "x2", "age_x1", "age_x2", "gender_x1", "gender_x2", "kinship_type", "age_diff"])

for i in range(n_rows):
    letter = random.choice(letters)
    x1 = f"{letter}{random.randint(1, 100)}"
    x2 = f"{letter}{random.randint(1, 100)}"
    kinship_type = random.choice(list(age_diff_distributions.keys()))
    gender_x1, gender_x2 = gender_constraints[kinship_type]
    if kinship_type == "sibs":
        gender_x1 = random.choice(["male", "female"])
        gender_x2 = "male" if gender_x1 == "female" else "female"
    distribution, *params = age_diff_distributions[kinship_type]
    age_diff = generate_age_diff(distribution, *params)
    age_x1 = random.randint(1, 100)
    age_x2 = max(0, int(age_x1 - age_diff)) if age_x1 >= age_diff else int(age_x1 + age_diff)
    if random.choice([True, False]):
        x1, x2 = x2, x1
        age_x1, age_x2 = age_x2, age_x1
        gender_x1, gender_x2 = gender_x2, gender_x1
    df_final.loc[i] = [x1, x2, age_x1, age_x2, gender_x1, gender_x2, kinship_type, math.ceil(age_diff)]

df_final

# %%
# Create df_pairwise
kinship_mapping = {
    "fs": "father-son",
    "fd": "father-daughter",
    "ms": "mother-son",
    "md": "mother-daughter",
    "bb": "brother-brother",
    "ss": "sister-sister",
    "sibs": "siblings",
    "gfgs": "grandfather-grandson",
    "gfgd": "grandfather-granddaughter",
    "gmgs": "grandmother-grandson",
    "gmgd": "grandmother-granddaughter",
}
rows = []
for (idx1, row1), (idx2, row2) in combinations(df_final.iterrows(), 2):
    age_diff = abs(row1["age_x1"] - row2["age_x1"])
    if row1["kinship_type"] == "sibs":
        first_term = "siblings"
    else:
        first_term = kinship_mapping[row1["kinship_type"]].split("-")[0]
    if row2["kinship_type"] == "sibs":
        second_term = "siblings"
    else:
        second_term = kinship_mapping[row2["kinship_type"]].split("-")[-1]
    new_kinship_type = first_term + "-" + second_term
    rows.append(
        {
            "x1": row1["x1"],
            "x2": row2["x1"],
            "age_x1": row1["age_x1"],
            "age_x2": row2["age_x1"],
            "gender_x1": row1["gender_x1"],
            "gender_x2": row2["gender_x1"],
            "age_diff": age_diff,
            "kinship_type": new_kinship_type,
        }
    )
df_pairwise = pd.DataFrame(rows)

# %%
# Add is_kin and age_diff columns
df_final["is_kin"] = 1
df_pairwise["is_kin"] = 0
df_pairwise.rename(columns={"new_kinship_type": "kinship_type"}, inplace=True)

# %%
# Merge to create df_merged
df_merged = pd.concat([df_final, df_pairwise], ignore_index=True)

# %%
df_merged.kinship_type.sort_values().unique()

# %%
