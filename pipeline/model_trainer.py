"""
Model Training Module
Implements all ML models described in the capstone proposal:
  - Linear Regression (+ Ridge, Lasso, ElasticNet)
  - Decision Tree
  - Random Forest (Bagging)
  - Gradient Boosting (Boosting)
  - Support Vector Regression
  - K-Nearest Neighbors
  - Multi-Layer Perceptron (ANN / Feed Forward Network)
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Model definitions with hyperparameter grids for tuning
MODEL_REGISTRY = {
    "linear": {
        "name": "Linear Regression",
        "class": LinearRegression,
        "params": {},
        "grid": {},
    },
    "ridge": {
        "name": "Ridge Regression",
        "class": Ridge,
        "params": {},
        "grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    },
    "lasso": {
        "name": "Lasso Regression",
        "class": Lasso,
        "params": {"max_iter": 10000},
        "grid": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },
    "elasticnet": {
        "name": "ElasticNet",
        "class": ElasticNet,
        "params": {"max_iter": 10000},
        "grid": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
    },
    "decision_tree": {
        "name": "Decision Tree",
        "class": DecisionTreeRegressor,
        "params": {},
        "grid": {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "random_forest": {
        "name": "Random Forest (Bagging)",
        "class": RandomForestRegressor,
        "params": {"n_jobs": -1},
        "grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
        },
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "class": GradientBoostingRegressor,
        "params": {},
        "grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "svr": {
        "name": "Support Vector Regression",
        "class": SVR,
        "params": {},
        "grid": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear", "poly"],
        },
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "class": KNeighborsRegressor,
        "params": {},
        "grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    },
    "mlp": {
        "name": "MLP Neural Network (ANN)",
        "class": MLPRegressor,
        "params": {"max_iter": 2000, "early_stopping": True, "random_state": 42},
        "grid": {
            "hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
        },
    },
}


class ModelTrainer:
    """Trains and tunes ML models for energy efficiency prediction."""

    def __init__(self, model_selection="all", random_state=42, cv_folds=5, tune=True):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.tune = tune

        if model_selection == "all":
            self.model_keys = list(MODEL_REGISTRY.keys())
        else:
            self.model_keys = [model_selection]

    def train(self, datasets: dict) -> dict:
        """
        Train all selected models on each target (heating/cooling).
        Returns nested dict: {target: {model_key: fitted_model}}
        """
        results = {}

        targets = [k for k in datasets.keys() if k not in ("scaler_X", "feature_names")]

        for target_name in targets:
            data = datasets[target_name]
            X_train = data["X_train"]
            y_train = data["y_train"]

            results[target_name] = {}

            for key in self.model_keys:
                spec = MODEL_REGISTRY[key]
                logger.info(f"  Training {spec['name']} for [{target_name}]...")

                # Build base model
                params = {**spec["params"]}
                if "random_state" in spec["class"]().get_params():
                    params["random_state"] = self.random_state

                model = spec["class"](**params)

                # Hyperparameter tuning with GridSearchCV
                if self.tune and spec["grid"]:
                    grid_search = GridSearchCV(
                        model, spec["grid"],
                        cv=self.cv_folds,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                        verbose=0,
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    logger.info(f"    Best params: {grid_search.best_params_}")
                    logger.info(f"    Best CV MSE: {-grid_search.best_score_:.4f}")
                else:
                    model.fit(X_train, y_train)

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.cv_folds,
                    scoring="neg_mean_absolute_error"
                )
                logger.info(f"    CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

                results[target_name][key] = model

        return results
