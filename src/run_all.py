import argparse
import json
import os
from pathlib import Path

from load_data import load_dataset
from experiment import run_experiment
from models import get_classifier_list


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATASETS = ["heart", "bank", "breast", "digits", "wine_quality"]
TRAIN_SPLITS = [0.2, 0.5, 0.8]
TRIALS = [1, 2, 3]

RAW_RESULTS_DIR = Path("results/raw")
AGG_RESULTS_DIR = Path("results/aggregated")

RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AGG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# RUN FULL GRID
# ---------------------------------------------------------

def run_full_experiment_matrix():
    """
    Runs:
       5 datasets × 3 splits × 3 classifiers × 3 trials = 135 runs
    """
    classifiers = get_classifier_list()

    for dataset_name in DATASETS:
        print(f"\n=== DATASET: {dataset_name.upper()} ===")

        # Load dataset once
        X, y = load_dataset(dataset_name)

        for split in TRAIN_SPLITS:
            print(f"  -> Train ratio: {split}")

            for clf_name, clf, param_grid in classifiers:
                print(f"     Classifier: {clf_name}")

                for trial in TRIALS:
                    print(f"        Trial {trial}")

                    results = run_experiment(
                        clf_name=clf_name,
                        clf=clf,
                        param_grid=param_grid,
                        X=X,
                        y=y,
                        train_ratio=split,
                        trial_id=trial,
                        dataset_name=dataset_name,
                    )

                    # Save raw JSON
                    out_file = RAW_RESULTS_DIR / f"{dataset_name}_{clf_name}_{split}_trial{trial}.json"
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=4)


# ---------------------------------------------------------
# AGGREGATE RESULTS
# ---------------------------------------------------------

def aggregate_results():
    """
    Reads every JSON file in results/raw/ and computes:
    mean + std for train, val, test accuracy
    grouped by (dataset, classifier, split)
    """
    import pandas as pd

    records = []

    for file in RAW_RESULTS_DIR.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)

        records.append({
            "dataset": data["dataset"],
            "classifier": data["classifier"],
            "split": data["train_ratio"],
            "trial": data["trial_id"],
            "train_acc": data["train_accuracy"],
            "val_acc": data["val_accuracy"],
            "test_acc": data["test_accuracy"]
        })

    df = pd.DataFrame(records)

    agg = df.groupby(["dataset", "classifier", "split"]).agg(
        mean_train_acc=("train_acc", "mean"),
        std_train_acc=("train_acc", "std"),
        mean_val_acc=("val_acc", "mean"),
        std_val_acc=("val_acc", "std"),
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"),
    ).reset_index()

    out_file = AGG_RESULTS_DIR / "aggregated_results.csv"
    agg.to_csv(out_file, index=False)

    print("\nAggregated results saved to:", out_file)


# ---------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true",
                        help="Run a quick experiment for ALL datasets.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate raw results into CSV.")

    args = parser.parse_args()

    # -----------------------------------------------------
    # TEST MODE — NOW TESTS ALL DATASETS
    # -----------------------------------------------------
    if args.test:
        print("Running TEST experiments (one classifier × all datasets)\n")

        clf_name, clf, param_grid = get_classifier_list()[0]  # SVM

        for dataset_name in DATASETS:
            print(f"\n--- TESTING {dataset_name.upper()} ---")

            X, y = load_dataset(dataset_name)

            results = run_experiment(
                clf_name=clf_name,
                clf=clf,
                param_grid=param_grid,
                X=X,
                y=y,
                train_ratio=0.2,
                trial_id=1,
                dataset_name=dataset_name,
            )

            print(json.dumps(results, indent=4))

    # -----------------------------------------------------
    # AGGREGATION MODE
    # -----------------------------------------------------
    elif args.aggregate:
        aggregate_results()

    # -----------------------------------------------------
    # FULL EXPERIMENT
    # -----------------------------------------------------
    else:
        total = len(DATASETS) * len(TRAIN_SPLITS) * len(get_classifier_list()) * len(TRIALS)
        print(f"Running FULL EXPERIMENT GRID ({total} runs)...")
        run_full_experiment_matrix()
        print("\nDONE! Now run:")
        print("   python src/run_all.py --aggregate")
