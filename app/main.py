import logging
import os
import time
from pydantic import BaseModel
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("heart_disease_api")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATHS = [
    os.path.join(MODELS_DIR, "best_model_pipeline.joblib"),
    os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
]

REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Total number of API requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "heart_disease_api_predictions_total",
    "Total number of prediction records processed",
)

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
    LOGGER.info("Loaded model from %s", model_path)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start
        status_code = str(response.status_code) if response is not None else "500"
        path = request.url.path
        method = request.method
        REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
        LOGGER.info(
            "request method=%s path=%s status_code=%s duration_ms=%.2f",
            method,
            path,
            status_code,
            duration * 1000,
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


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
        PREDICTION_COUNT.inc(len(df))
        LOGGER.info("prediction batch_size=%s", len(df))
        return {"predictions": preds.tolist(), "probabilities": probs}
    except Exception as e:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
