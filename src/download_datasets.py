# src/download_datasets.py
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)


def _to_series(y) -> pd.Series:
    """Flatten y to a 1D pandas Series."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            y = y.iloc[:, 0]
    return pd.Series(y)


def binarize_target(name: str, y) -> pd.Series:
    """
    Convert raw UCI targets to binary labels y in {0,1} according to the project plan.
    """
    y = _to_series(y)

    if name == "bank":
        y_str = y.astype(str).str.strip().str.lower()
        return (y_str == "yes").astype(int)

    if name == "breast":
        y_str = y.astype(str).str.strip().str.upper()
        return (y_str.isin(["M", "MALIGNANT", "1"])).astype(int)

    if name == "heart":
        y_num = pd.to_numeric(y, errors="coerce")
        return (y_num > 0).astype(int)

    if name == "digits":
        y_num = pd.to_numeric(y, errors="coerce")
        return (y_num == 0).astype(int)

    if name == "wine_quality":
        y_num = pd.to_numeric(y, errors="coerce")
        return (y_num >= 6).astype(int)

    if y.dtype == "object":
        return pd.Series(pd.factorize(y)[0])
    return y


def save_dataset(name: str, data, filename: str):
    X = data.data.features
    y = data.data.targets
    y_bin = binarize_target(name, y)
    df = X.copy()
    df["target"] = y_bin
    out_path = os.path.join(SAVE_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  (target is binary: {sorted(df['target'].unique())})")
