from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "random_forest_pipeline.joblib")

app = FastAPI(title="Heart Disease Predictor")


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


@app.on_event("startup")
def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)


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
