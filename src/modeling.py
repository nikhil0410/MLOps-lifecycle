"""Shared modeling configuration for training and export flows."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TARGET_COLUMN = "target"


def build_model_candidates(random_state: int = RANDOM_STATE):
    """Return model estimators with small, reproducible tuning grids."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=random_state,
            ),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__class_weight": [None, "balanced"],
            },
        },
    }
