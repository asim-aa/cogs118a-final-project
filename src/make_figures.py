# python src/make_figures.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

sns.set(style="whitegrid")

RESULTS_FILE = "results/aggregated/aggregated_results.csv"
RAW_DIR = Path("results/raw")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)
df["split"] = pd.to_numeric(df["split"])
df = df.sort_values("split")

dataset_order = sorted(df["dataset"].unique())
classifier_order = ["svm", "rf", "mlp", "knn"]

df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
df["classifier"] = pd.Categorical(df["classifier"], categories=classifier_order, ordered=True)

# ====================================================
# FIGURE 1 — Bar plot: Mean test accuracy
# ====================================================
plt.figure(figsize=(12, 7))
sns.barplot(data=df, x="classifier", y="mean_test_acc", hue="dataset")
plt.title("Mean Test Accuracy by Classifier and Dataset")
plt.savefig(OUTDIR / "test_accuracy_by_classifier_dataset.png")
plt.close()

# ====================================================
# FIGURE 2 — Line plot: training split effect
# ====================================================
plt.figure(figsize=(12, 7))
sns.lineplot(data=df, x="split", y="mean_test_acc", hue="classifier", style="dataset", markers=True)
plt.title("Accuracy vs Training Split")
plt.savefig(OUTDIR / "accuracy_vs_split.png")
plt.close()

# ====================================================
# FIGURE 3 — Error bars
# ====================================================
plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=df, x="classifier", y="mean_test_acc",
    hue="dataset"
)
# Overlay correct std error bars using precomputed std_test_acc
for i, bar in enumerate(ax.patches):
    # bars are grouped by hue; compute matching std index
    mean = bar.get_height()
    std = df.iloc[i]["std_test_acc"]
    x = bar.get_x() + bar.get_width() / 2
    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="black", capsize=4, lw=1)
plt.title("Test Accuracy: Mean ± Standard Deviation")
plt.savefig(OUTDIR / "test_accuracy_error_bars.png")
plt.close()

# ====================================================
# FIGURE 4 — Heatmap of classifier performance
# ====================================================
pivot = df.pivot_table(index="dataset", columns="classifier", values="mean_test_acc")
pivot = pivot[classifier_order]
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
plt.title("Classifier Performance Heatmap")
plt.savefig(OUTDIR / "performance_heatmap.png")
plt.close()

# ====================================================
# FIGURE 5 — Hyperparameter visualization
# ====================================================
hyper_records = []

for file in RAW_DIR.glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
    params = dict(data["best_params"])  # copy to avoid mutating the source dict
    params["classifier"] = data["classifier"]
    params["dataset"] = data["dataset"]
    params["val_acc"] = data["val_accuracy"]
    hyper_records.append(params)

hp_df = pd.DataFrame(hyper_records)

# SVM hyperparameters
if "C" in hp_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=hp_df, x="C", y="val_acc", hue="dataset")
    plt.title("SVM: Validation Accuracy vs C")
    plt.savefig(OUTDIR / "svm_hyperparam_C.png")
    plt.close()

if "gamma" in hp_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=hp_df, x="gamma", y="val_acc", hue="dataset")
    plt.title("SVM: Validation Accuracy vs Gamma")
    plt.savefig(OUTDIR / "svm_hyperparam_gamma.png")
    plt.close()

# RF hyperparameters
if "n_estimators" in hp_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=hp_df, x="n_estimators", y="val_acc", hue="dataset")
    plt.title("RF: Validation Accuracy vs n_estimators")
    plt.savefig(OUTDIR / "rf_hyperparam_estimators.png")
    plt.close()

# MLP hyperparameters
if "learning_rate_init" in hp_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=hp_df, x="learning_rate_init", y="val_acc", hue="dataset")
    plt.title("MLP: Validation Accuracy vs Learning Rate")
    plt.savefig(OUTDIR / "mlp_hyperparam_lr.png")
    plt.close()

# ====================================================
# FIGURE 6A — Training Accuracy vs Split
# ====================================================
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df,
    x="split", y="mean_train_acc",
    hue="classifier", style="dataset", markers=True
)
plt.title("Training Accuracy vs Training Split")
plt.savefig(OUTDIR / "train_accuracy.png")
plt.close()

# ====================================================
# FIGURE 6B — Validation Accuracy vs Split
# ====================================================
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df,
    x="split", y="mean_val_acc",
    hue="classifier", style="dataset", markers=True
)
plt.title("Validation Accuracy vs Training Split")
plt.savefig(OUTDIR / "validation_accuracy.png")
plt.close()

# ====================================================
# FIGURE 6C — Train–Validation Gap
# ====================================================
df["train_val_gap"] = df["mean_train_acc"] - df["mean_val_acc"]

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df,
    x="split", y="train_val_gap",
    hue="classifier", style="dataset", markers=True
)
plt.title("Overfitting Gap (Train - Validation Accuracy)")
plt.ylabel("Gap")
plt.savefig(OUTDIR / "train_val_gap.png")
plt.close()
