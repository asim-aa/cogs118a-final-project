from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def load_dataset(name):
    """
    Loads and preprocesses datasets:
        - heart
        - bank
        - breast
        
    Steps:
        1. Load
        2. Impute numeric (median)
        3. One-hot encode categoricals
        4. Standardize numeric
    """

    # -------------------------------------------------
    # LOAD RAW
    # -------------------------------------------------
    if name == "heart":
        data = fetch_ucirepo(id=45)
        X = data.data.features.copy()
        y = data.data.targets["num"].apply(lambda v: 1 if v > 0 else 0)

    elif name == "bank":
        data = fetch_ucirepo(id=222)
        X = data.data.features.copy()
        y = data.data.targets["y"].map({"yes": 1, "no": 0})

    elif name == "breast":
        data = fetch_ucirepo(id=17)
        X = data.data.features.copy()
        y = data.data.targets["Diagnosis"].map({"M": 1, "B": 0})

    else:
        raise ValueError(f"Unknown dataset: {name}")


    # -------------------------------------------------
    # IDENTIFY COLUMN TYPES
    # -------------------------------------------------
    numeric_cols = X.select_dtypes(include=["int", "float"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # -------------------------------------------------
    # BUILD PREPROCESSOR
    # -------------------------------------------------
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

    # -------------------------------------------------
    # FIT + TRANSFORM
    # -------------------------------------------------
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y
