import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")

RESULTS_FILE = "results/aggregated/aggregated_results.csv"
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

# 1. BAR PLOT — Mean test accuracy per classifier per dataset
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="classifier", y="mean_test_acc", hue="dataset")
plt.title("Mean Test Accuracy by Classifier and Dataset")
plt.ylabel("Accuracy")
plt.savefig(OUTDIR / "test_accuracy_by_classifier_dataset.png")
plt.close()

# 2. LINE PLOT — Accuracy vs. train split
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="split", y="mean_test_acc", hue="classifier", style="dataset", markers=True)
plt.title("Accuracy vs. Training Split")
plt.ylabel("Accuracy")
plt.savefig(OUTDIR / "accuracy_vs_split.png")
plt.close()

# 3. ERROR BARS (mean ± std)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x="classifier", y="mean_test_acc",
    hue="dataset",
    capsize=0.2,
    errorbar=("sd")
)
plt.title("Test Accuracy Mean ± Std")
plt.ylabel("Accuracy")
plt.savefig(OUTDIR / "test_accuracy_error_bars.png")
plt.close()

# 4. HEATMAP — Classifier performance matrix
pivot = df.pivot_table(index="dataset", columns="classifier", values="mean_test_acc")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
plt.title("Classifier Performance Heatmap (Test Accuracy)")
plt.savefig(OUTDIR / "performance_heatmap.png")
plt.close()

print("All plots saved in results/figures/")
