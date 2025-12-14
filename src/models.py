from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------
# Return a specific classifier instance
# ---------------------------------------------------------

def get_classifier(model_name, seed=None):
    if model_name == "svm":
        # class_weight helps for imbalanced datasets (e.g., bank)
        return SVC(
            kernel="rbf",
            probability=False,
            class_weight="balanced",
            cache_size=1024,
            random_state=seed
        )

    if model_name == "rf":
        # balanced_subsample is a good default for imbalanced data
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    if model_name == "mlp":
        # early_stopping improves stability/convergence and reduces wasted compute
        return MLPClassifier(
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=seed
        )

    if model_name == "knn":
        return KNeighborsClassifier()

    raise ValueError(f"Unknown classifier: {model_name}")


# ---------------------------------------------------------
# Return hyperparameter grid for GridSearchCV
# ---------------------------------------------------------

def get_param_grid(model_name):

    if model_name == "svm":
        # modest but real tuning; still small enough to run
        return {
            "C": [0.1, 1, 10],
            "gamma": [0.01, 0.1, "scale"],
            "kernel": ["rbf"],
        }

    if model_name == "rf":
        # modest grid; gives a meaningful search without blowing up runtime
        return {
            "n_estimators": [200, 500],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt"],
        }

    if model_name == "mlp":
        # adds a bit more breadth for tuning while staying manageable
        return {
            "hidden_layer_sizes": [(64,), (128,), (64, 64)],
            "learning_rate_init": [0.001, 0.01],
            "alpha": [0.0001, 0.001],
            "activation": ["relu"]
        }

    if model_name == "knn":
        # include p=1 vs p=2 (Manhattan vs Euclidean)
        return {
            "n_neighbors": [3, 5, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }

    raise ValueError(f"No param grid for {model_name}")


# ---------------------------------------------------------
# List of all classifiers to run in run_all.py
# ---------------------------------------------------------

def get_classifier_list():
    return [
        ("svm", get_param_grid("svm")),
        ("rf", get_param_grid("rf")),
        ("mlp", get_param_grid("mlp")),
        ("knn", get_param_grid("knn")),
    ]
