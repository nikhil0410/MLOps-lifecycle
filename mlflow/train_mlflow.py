import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

DATA_PATH = os.path.join("heart_disease", "processed.cleveland.data")

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(COLUMN_NAMES):
        df.columns = COLUMN_NAMES
    else:
        # fallback: assign generic names
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        df = df.rename(columns={df.columns[-1]: "target"})
    # replace missing markers and coerce to numeric
    df = df.replace("?", np.nan)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # convert target to binary (0 = no disease, 1 = disease)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df


def build_pipeline():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    return pipe


def main():
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()

    with mlflow.start_run() as run:
        mlflow.log_param("candidate_model", "random_forest")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:, 1]

        # Log a small artifact: sample predictions
        sample = X_val.copy()
        sample["y_true"] = y_val.values
        sample["y_pred"] = preds
        sample["y_proba"] = probs
        sample_path = "sample_predictions.csv"
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)

        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipe.predict(input_example))
        mlflow.sklearn.log_model(
            pipe,
            name="random_forest_pipeline",
            input_example=input_example,
            signature=signature,
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id
        print(f"MLflow run completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
