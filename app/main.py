from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATHS = [
    os.path.join(MODELS_DIR, "best_model_pipeline.joblib"),
    os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
]

app = FastAPI(title="Heart Disease Predictor")


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


@app.on_event("startup")
def load_model():
    global MODEL
    model_path = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
    if model_path is None:
        raise RuntimeError(f"Model file not found. Checked: {MODEL_PATHS}")
    MODEL = joblib.load(model_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        df = pd.DataFrame(payload.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No instances provided")

    try:
        preds = MODEL.predict(df)
        probs = None
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(df)[:, 1].tolist()
        return {"predictions": preds.tolist(), "probabilities": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
