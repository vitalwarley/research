# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check datasets: RFIW2020 vs RFIW2021

# %% [markdown]
# ## Introduction

# %% [markdown]
# This notebook was created by [Jupyter AI](https://github.com/jupyterlab/jupyter-ai) with the following prompt:
#
# > /generate check if folders in ../fitw2020/train-faces-det are equal to those in ../rfiw2021/Track1/Train/train-faces

# %% [markdown]
# This Jupyter notebook focuses on verifying if the contents of two specific directories, namely, '../fitw2020/train-faces-det' and '../rfiw2021/Track1/Train/train-faces' are identical or not. The process is initiated by importing Python's 'os' library to interact with the operating system. It then defines the paths to the directories in question, which are assigned to variables 'rfiw2020' and 'rfiw2021'. Subsequently, two lists are created that hold the names of the files and subdirectories in 'rfiw2020' and 'rfiw2021g', respectively, through the 'os.listdir' function. A comparison operation is then performed on the contents of these lists using the '==' operator, validating whether the lists (and hence the directories) are equal or not. The final step includes printing the result of this comparison, indicating the equality or disparity between the directories.

# %% [markdown]
# ## Define Directory Paths

# %%
import os


# %%
def get_absolute_path(relative_path):
    return os.path.abspath(relative_path)


# %%
rfiw2020 = get_absolute_path("../fitw2020/train-faces-det")
rfiw2021 = get_absolute_path("../rfiw2021/Track1/Train/train-faces")

# %% [markdown]
# ## Retrieve Folder Contents

# %%
rfiw2020_contents = os.listdir(rfiw2020)
rfiw2021_contents = os.listdir(rfiw2021)


# %% [markdown]
# ## Compare Folder Contents


# %%
def compare_directories(rfiw2020, rfiw2021):
    rfiw2020_contents = set(os.listdir(rfiw2020))
    rfiw2021_contents = set(os.listdir(rfiw2021))
    if rfiw2020_contents == rfiw2021_contents:
        print("The directories are equal.")
    else:
        rfiw2021_complement = rfiw2020_contents - rfiw2021_contents
        print(f"Only in rfiw2021: {rfiw2021_complement}")
        rfiw2020_complement = rfiw2021_contents - rfiw2020_contents
        print(f"Only in rfiw2020: {rfiw2020_complement}")
        print("The directories are not equal.")


# %%
compare_directories(rfiw2020, rfiw2021)
