import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from load_data import load_dataset
from experiment import run_experiment
from models import get_classifier_list

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATASETS = ["heart", "bank", "breast"]
TRAIN_SPLITS = [0.2, 0.5, 0.8]    # 20/80, 50/50, 80/20
TRIALS = [1, 2, 3]                # three runs per config

RAW_RESULTS_DIR = Path("results/raw")
AGG_RESULTS_DIR = Path("results/aggregated")

RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AGG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# RUN ALL EXPERIMENTS
# ---------------------------------------------------------

def run_full_experiment_matrix():
    """
    Runs the full grid:
        3 datasets × 3 splits × 3 classifiers × 3 trials = 81 runs.
    Saves raw JSON logs for each run.
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
                        dataset_name=dataset_name
                    )
                    

                    # save raw JSON
                    out_file = RAW_RESULTS_DIR / f"{dataset_name}_{clf_name}_{split}_trial{trial}.json"
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=4)


# ---------------------------------------------------------
# AGGREGATE RESULTS
# ---------------------------------------------------------

def aggregate_results():
    """
    Reads every JSON file in results/raw/ and computes
    mean + std test accuracy grouped by:
        (dataset, classifier, split)
    """
    import pandas as pd
    import numpy as np

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
                        help="Run only 1 quick experiment for debugging.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate raw results into CSV.")

    args = parser.parse_args()

    if args.test:
        print("Running TEST experiment (1 dataset × 1 classifier × 1 split × 1 trial)")

        X, y = load_dataset("heart")  # smallest dataset
        clf_name, clf, param_grid = get_classifier_list()[0]

        results = run_experiment(
            clf_name=clf_name,
            clf=clf,
            param_grid=param_grid,
            X=X,
            y=y,
            train_ratio=0.2,
            trial_id=1,
            dataset_name="heart"
        )

        print("\nTEST RUN RESULTS:\n", json.dumps(results, indent=4))

    elif args.aggregate:
        aggregate_results()

    else:
        print("Running FULL EXPERIMENT GRID (81 runs)...")
        run_full_experiment_matrix()
        print("\nDONE! Now run:")
        print("   python src/run_all.py --aggregate")
