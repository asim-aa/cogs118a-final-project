import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

warnings.filterwarnings("ignore")


def run_experiment(clf_name, clf, param_grid, X, y, train_ratio, trial_id,dataset_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_ratio,
        stratify=y,
        random_state=trial_id
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_id)

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    best_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    val_acc = grid.best_score_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    return {
        
        "dataset": dataset_name,
        "classifier": clf_name,
        "train_ratio": train_ratio,
        "trial_id": trial_id,
        "best_params": best_params,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc)
    }
