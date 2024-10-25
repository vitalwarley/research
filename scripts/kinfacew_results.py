import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv("kinfacew_results.csv")

# Rename kinship types (naming error in logging): Sample.NAME2LABEL : KinFaceWDataset labels
rename_kin = {"accuracy/non-kin": "fd", "accuracy/md": "fs", "accuracy/ms": "md", "accuracy/sibs": "ms"}
df = df.rename(columns=rename_kin)
df.columns = [
    "run",
    "label",
    "batch_size",
    "dataset",
    "fold",
    "lr",
    "alpha",
    "tau",
    "accuracy",
    "auc",
    "fd",
    "fs",
    "md",
    "ms",
]

# Split
df_cross_eval = df.iloc[-2:, :]
df = df.iloc[:-2, :]


# Group by fold and compute mean accuracy and AUC
result = (
    df.groupby(["dataset", "batch_size", "lr"])
    .agg(
        {
            "accuracy": "mean",
            "auc": "mean",
            "fd": "mean",
            "fs": "mean",
            "md": "mean",
            "ms": "mean",
        }
    )
    .reset_index()
)

# Print results
print("Mean Accuracy and AUC by Dataset:")
print(result.to_string(index=False))

# Optionally, save results to a CSV file
result.to_csv("mean_metrics_by_dataset.csv", index=False)
