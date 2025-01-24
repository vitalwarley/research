from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.read_csv("datasets/facornet/txt/train_sort_A2_m.txt", delimiter=" ", header=None)
data.columns = ["ID", "F1", "F2", "Kinship", "IsKin", "Confidence"]
# Drop ID and Confidence columns
data = data.drop(columns=["ID", "Confidence"])

# Extract and store non-symmetrical pairs
non_symmetrical_identifiers = []
# Extract and store symmetrical pairs
symmetrical_identifiers = []

for row in data.itertuples():
    parts1 = row.F1.split("/")
    parts2 = row.F2.split("/")
    f1_id = parts1[2] + "/" + parts1[3]
    f2_id = parts2[2] + "/" + parts2[3]

    non_symmetrical_identifiers.append((f1_id, f2_id))  # Keep original order
    symmetrical_identifiers.append(tuple(sorted([f1_id, f2_id])))  # Sort the pair for symmetry

# Count occurrences
non_symmetrical_counts = Counter(non_symmetrical_identifiers)
symmetrical_counts = Counter(symmetrical_identifiers)

# Create DataFrames for plotting
df_non_symmetrical = pd.DataFrame(non_symmetrical_counts.items(), columns=["Identifier", "Frequency"])
df_non_symmetrical["Identifier"] = pd.factorize(df_non_symmetrical["Identifier"])[0]

df_symmetrical = pd.DataFrame(symmetrical_counts.items(), columns=["Identifier", "Frequency"])
df_symmetrical["Identifier"] = pd.factorize(df_symmetrical["Identifier"])[0]

# Plot both cases
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 20))

# Non-Symmetrical
axes[0].bar(df_non_symmetrical["Identifier"], df_non_symmetrical["Frequency"])
axes[0].set_xlabel("Individuals")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Non-Symmetrical Pairs")
axes[0].tick_params(axis="x", rotation=90)

# Symmetrical
axes[1].bar(df_symmetrical["Identifier"], df_symmetrical["Frequency"])
axes[1].set_xlabel("Individuals")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Symmetrical Pairs")
axes[1].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# Convert counts to DataFrame for easier manipulation
df_non_sym = pd.DataFrame.from_records(list(non_symmetrical_counts.items()), columns=["Pair", "NonSymFreq"])
df_sym = pd.DataFrame.from_records(list(symmetrical_counts.items()), columns=["Pair", "SymFreq"])

# Merge on Pair to compare frequencies directly (this is a rough comparison)
df_merged = df_non_sym.merge(df_sym, how="left", on="Pair")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_merged["NonSymFreq"], df_merged["SymFreq"], alpha=0.5)
plt.title("Comparison of Non-Symmetrical vs Symmetrical Frequencies")
plt.xlabel("Non-Symmetrical Frequency")
plt.ylabel("Symmetrical Frequency")
plt.grid(True)
plt.show()
