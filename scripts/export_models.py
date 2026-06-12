"""Train final pipelines and export them to models/ as joblib files.
This creates reproducible model artifacts usable by the serving app.
"""
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.preprocessing import build_preprocessor

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(ROOT, "data", "processed_cleveland.csv")
OUT_DIR = os.path.join(ROOT, "models")

def ensure():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["target"]) if "target" in df.columns else df.iloc[:, :-1]
    y = df["target"] if "target" in df.columns else df.iloc[:, -1]
    return X, y

def main():
    ensure()
    X, y = load_data()
    pre = build_preprocessor(pd.concat([X, y], axis=1))

    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X, y)
        out_path = os.path.join(OUT_DIR, f"{name}_pipeline.joblib")
        joblib.dump(pipe, out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
