import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV
df = pd.read_csv("tensorboard_data.csv")
df["normalize"] = df["normalize"].astype("category")

# Filter data for 'epoch_auc' metric
filtered_df = df[(df["metric"] == "epoch_auc") & (df["step"] < 6000)]

# Find the max 'epoch_auc' for each experiment/session before step 6000
max_epoch_auc = filtered_df.groupby("experiment")["value"].max()

# Sort and get top 10 experiments
top_10_experiments = max_epoch_auc.sort_values(ascending=False).head(10)

# Print batch-size, clip-gradient, normalize, and num-epoch for each experiment
experiments = []
for experiment in top_10_experiments.index:
    exp = df[df["experiment"] == experiment][
        ["experiment", "batch-size", "clip-gradient", "normalize", "num-epoch", "metric", "value"]
    ]
    exp["experiment"] = experiment
    experiments.append(exp)

# Concatenate the experiments into a DataFrame
experiments_df = pd.concat(experiments)

# Plot a scatter plot between batch-size, clip-gradient, normalize, and num-epoch vs epoch_auc
metric = "epoch_auc"
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
for idx, col in enumerate(["batch-size", "clip-gradient", "normalize", "num-epoch"]):
    metric_df = experiments_df[experiments_df.metric == metric]
    axes[idx].scatter(metric_df[col], metric_df["value"])
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("epoch_auc")
    axes[idx].set_title(f"{col} vs epoch_auc")
    # Make xticks No and Yes for normalize
    if col == "normalize":
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["No", "Yes"])


plt.savefig("epoch_auc_vs_hparams.png")
plt.show()
