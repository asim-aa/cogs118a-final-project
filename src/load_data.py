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

    # ----------------------------
    # Subsample BANK to speed up runtime (stratified)
    # ----------------------------
    if name == "bank" and len(df) > 15000:
        N = 15000
        df = df.groupby("target", group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, int(N * len(g) / len(df))),
                random_state=0
            )
        )

    y = df["target"]
    X = df.drop(columns=["target"])

    # Ensure labels are numeric (important for MLP + sklearn internals)
    if y.dtype == "object":
        y = pd.factorize(y)[0]  # maps strings to {0,1,2,...}

    return X, y
