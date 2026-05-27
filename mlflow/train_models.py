"""Train multiple models with a preprocessing pipeline, CV, and MLflow logging.
Trains LogisticRegression and RandomForest, logs CV scores and final models.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import sys
import os
# ensure project root is on sys.path when running this script from the mlflow/ folder
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.preprocessing import build_preprocessor

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(ROOT, "data", "processed_cleveland.csv")
OUT = os.path.join(ROOT, "artifacts", "models")

def ensure():
    os.makedirs(OUT, exist_ok=True)

def load_data():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def cv_and_log(name, pipeline, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False)
    df_scores = pd.DataFrame({k: v for k, v in scores.items()})
    scores_path = os.path.join(OUT, f"cv_{name}.csv")
    df_scores.to_csv(scores_path, index=False)
    mlflow.log_artifact(scores_path)
    return df_scores

def main():
    ensure()
    mlflow.sklearn.autolog()
    X, y = load_data()
    pre = build_preprocessor(pd.concat([X, y], axis=1))

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])
        with mlflow.start_run(run_name=name):
            df_scores = cv_and_log(name, pipe, X, y)
            # fit on full data and log model
            pipe.fit(X, y)
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_{name}")
            print(f"Completed run for {name}. CV scores saved to artifacts and logged to MLflow.")

if __name__ == "__main__":
    main()
