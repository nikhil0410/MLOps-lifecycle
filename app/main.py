import logging
import os
import time
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

try:
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    MlflowClient = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("heart_disease_api")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MLFLOW_DB = os.path.join(ROOT_DIR, "mlflow.db")
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


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_MLFLOW_DB}")


def get_mlflow_client() -> MlflowClient:
    if MlflowClient is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow is not installed in the serving environment.",
        )
    return MlflowClient(tracking_uri=get_tracking_uri())


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


@app.get("/mlflow")
def mlflow_summary():
    client = get_mlflow_client()
    try:
        experiments = client.search_experiments()
        summary = []
        for experiment in experiments:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=5,
            )
            summary.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                    "latest_runs": [
                        {
                            "run_id": run.info.run_id,
                            "run_name": run.data.tags.get("mlflow.runName"),
                            "status": run.info.status,
                            "start_time": run.info.start_time,
                            "metrics": run.data.metrics,
                            "params": run.data.params,
                        }
                        for run in runs
                    ],
                }
            )
    except Exception as exc:
        LOGGER.exception("MLflow summary lookup failed")
        raise HTTPException(status_code=500, detail=f"Unable to query MLflow: {exc}")

    return {
        "tracking_uri": get_tracking_uri(),
        "experiment_count": len(summary),
        "experiments": summary,
    }


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
