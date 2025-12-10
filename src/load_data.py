import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(name):
    """
    Loads and preprocesses datasets from local CSVs.
    Expects files:
        data/heart.csv
        data/bank.csv
        data/breast.csv

    Each CSV MUST contain a 'target' column.
    """

    path = f"data/{name}.csv"

    # ----------------------------
    # Load local CSV
    # ----------------------------
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(f"'target' column missing in {path}")

    y = df["target"]
    X = df.drop(columns=["target"])

    # ----------------------------
    # Identify column types
    # ----------------------------
    numeric_cols = X.select_dtypes(include=["int", "float"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # ----------------------------
    # Preprocessing pipelines
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # ----------------------------
    # Fit + transform
    # ----------------------------
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y
