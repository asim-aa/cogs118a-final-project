import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


def _build_preprocessor(X):
    # Robust dtype detection
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if len(transformers) == 0:
        raise ValueError("No usable feature columns found (numeric or categorical).")

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def run_experiment(clf_name, clf_factory, param_grid, X, y, train_ratio, trial_id, dataset_name):
    seed = trial_id
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_ratio,
        stratify=y,
        random_state=seed
    )

    # Build preprocessor using TRAIN ONLY (still safe; actual fitting happens inside CV folds)
    preprocessor = _build_preprocessor(X_train)
    clf = clf_factory(seed)

    # Prefix param grid for the pipeline
    pipeline_param_grid = {f"clf__{k}": v for k, v in param_grid.items()}

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"  
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    # Drop pipeline prefix for logging
    best_params_clean = {
        (k.replace("clf__", "", 1) if k.startswith("clf__") else k): v
        for k, v in best_params.items()
    }

    best_cv_score = grid.best_score_
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    return {
        "dataset": dataset_name,
        "classifier": clf_name,
        "train_ratio": train_ratio,
        "trial_id": trial_id,
        "seed": seed,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "best_params": best_params_clean,
        "best_cv_score": float(best_cv_score),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(best_cv_score),
        "test_accuracy": float(test_acc)
    }
