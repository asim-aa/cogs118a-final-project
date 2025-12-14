# python src/make_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")
RESULTS_FILE = "results/aggregated/aggregated_results.csv"
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RESULTS_FILE)
df["split"] = pd.to_numeric(df["split"])
df = df.sort_values(["dataset", "split", "classifier"]).reset_index(drop=True)
dataset_order = sorted(df["dataset"].unique())
classifier_order = ["svm", "rf", "mlp", "knn"]  # keep consistent with your experiments
split_order = sorted(df["split"].unique())
df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
df["classifier"] = pd.Categorical(df["classifier"], categories=classifier_order, ordered=True)
df["split"] = pd.Categorical(df["split"], categories=split_order, ordered=True)
df_out = df[["dataset", "split", "classifier", "mean_test_acc", "std_test_acc"]].copy()
df_out.to_csv(OUTDIR / "table_dataset_split_classifier_testacc.csv", index=False)

g = sns.catplot(
    data=df,
    x="classifier",
    y="mean_test_acc",
    row="dataset",
    col="split",
    kind="bar",
    order=classifier_order,
    row_order=dataset_order,
    col_order=split_order,
    height=2.2,
    aspect=1.2,
    sharey=True
)

for ax in g.axes.flatten():
    if ax is None:
        continue

    split_val = float(ax.get_title().split("=")[-1].strip())
    row_i = list(g.axes[:, 0]).index(ax) if ax in list(g.axes[:, 0]) else None

   
    for r in range(g.axes.shape[0]):
        for c in range(g.axes.shape[1]):
            if g.axes[r, c] is ax:
                dataset_val = dataset_order[r]
                split_val = split_order[c]
                subset = df[(df["dataset"] == dataset_val) & (df["split"] == split_val)].copy()
                subset = subset.sort_values("classifier")
                # One bar per classifier, patches are in classifier order
                for patch, (_, row) in zip(ax.patches, subset.iterrows()):
                    mean = row["mean_test_acc"]
                    std = row.get("std_test_acc", 0.0)
                    x = patch.get_x() + patch.get_width() / 2
                    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="black", capsize=2, lw=0.8)
                break

g.set_axis_labels("Classifier", "Mean Test Accuracy")
g.fig.suptitle("Algorithm Comparison per Dataset and Train/Test Partition", y=1.02)
g.savefig(OUTDIR / "rubric_a_dataset_by_split_algorithm.png", dpi=200, bbox_inches="tight")
plt.close(g.fig)

g2 = sns.relplot(
    data=df,
    x="split",
    y="mean_test_acc",
    col="classifier",
    hue="dataset",
    kind="line",
    marker="o",
    col_order=classifier_order,
    hue_order=dataset_order,
    height=3.0,
    aspect=1.1
)

g2.set_axis_labels("Training Split (train ratio)", "Mean Test Accuracy")
g2.fig.suptitle("Effect of Training Split per Classifier", y=1.05)
g2.savefig(OUTDIR / "rubric_b_classifier_trend_across_splits.png", dpi=200, bbox_inches="tight")
plt.close(g2.fig)

print("Saved figures to:", OUTDIR)
print(" - rubric_a_dataset_by_split_algorithm.png")
print(" - rubric_b_classifier_trend_across_splits.png")
print(" - table_dataset_split_classifier_testacc.csv (optional table)")
