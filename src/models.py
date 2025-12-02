from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------
# Return a specific classifier instance
# ---------------------------------------------------------

def get_classifier(model_name):
    if model_name == "svm":
        return SVC()
    
    if model_name == "rf":
        return RandomForestClassifier()
    
    if model_name == "mlp":
        return MLPClassifier(max_iter=500)
    
    raise ValueError(f"Unknown classifier: {model_name}")


# ---------------------------------------------------------
# Return hyperparameter grid for GridSearchCV
# ---------------------------------------------------------

def get_param_grid(model_name):

    if model_name == "svm":
        return {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"]
        }
    
    if model_name == "rf":
        return {
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 4]
        }
    
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "learning_rate_init": [0.001, 0.01],
            "activation": ["relu"]
        }
    
    raise ValueError(f"No param grid for {model_name}")


# ---------------------------------------------------------
# List of all classifiers to run in run_all.py
# ---------------------------------------------------------

def get_classifier_list():
    return [
        ("svm", SVC(), get_param_grid("svm")),
        ("rf", RandomForestClassifier(), get_param_grid("rf")),
        ("mlp", MLPClassifier(max_iter=500), get_param_grid("mlp")),
    ]
