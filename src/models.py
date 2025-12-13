from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------
# Return a specific classifier instance
# ---------------------------------------------------------

def get_classifier(model_name, seed=None):
    if model_name == "svm":
        return SVC(kernel="rbf", probability=False, random_state=seed)
    
    if model_name == "rf":
        return RandomForestClassifier(random_state=seed)
    
    if model_name == "mlp":
        return MLPClassifier(max_iter=500, random_state=seed)

    if model_name == "knn":
        return KNeighborsClassifier()
    
    raise ValueError(f"Unknown classifier: {model_name}")


# ---------------------------------------------------------
# Return hyperparameter grid for GridSearchCV
# ---------------------------------------------------------

def get_param_grid(model_name):

    if model_name == "svm":
        return {
            "C": [1, 10],
            "gamma": [0.01, "scale"],
            "kernel": ["rbf"],
        }
        
    if model_name == "rf":
        return {
            "n_estimators": [200],
            "max_depth": [None, 20],
            "min_samples_split": [2],
            "max_features": ["sqrt"],
        }
    
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": [(64,), (128,)],
            "learning_rate_init": [0.001],
            "alpha": [0.0001, 0.001],
            "activation": ["relu"]
        }

    if model_name == "knn":
        return {
            "n_neighbors": [5, 9],
            "weights": ["uniform", "distance"],
            "p": [2],
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
