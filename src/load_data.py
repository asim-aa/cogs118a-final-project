import pandas as pd
import os


def load_dataset(name):
    """
    Loads and preprocesses datasets from local CSVs.

    Supported dataset names:
        "heart"
        "bank"
        "breast"
        "digits"
        "wine_quality"

    Each CSV MUST contain a 'target' column.
    """

    path = f"data/{name}.csv"

    # ----------------------------
    # Check file existence
    # ----------------------------
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Make sure you ran download_datasets.py first."
        )

    # ----------------------------
    # Load local CSV
    # ----------------------------
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(f"'target' column missing in {path}")

    y = df["target"]
    X = df.drop(columns=["target"])

    # Return raw features/labels; preprocessing is applied inside pipelines
    # within the experiment to avoid leakage.
    return X, y
