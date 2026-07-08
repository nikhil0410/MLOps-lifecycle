from fastapi.testclient import TestClient
import numpy as np

import app.main as app_main


class DummyModel:
    def predict(self, df):
        return np.array([1] * len(df))

    def predict_proba(self, df):
        return np.array([[0.2, 0.8] for _ in range(len(df))])


def test_health_endpoint():
    client = TestClient(app_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_predictions(monkeypatch):
    monkeypatch.setattr(app_main, "MODEL", DummyModel(), raising=False)
    monkeypatch.setattr(app_main, "MODEL_PATHS", ["dummy-model-path.joblib"], raising=False)
    monkeypatch.setattr(app_main.os.path, "exists", lambda _: True)
    monkeypatch.setattr(app_main.joblib, "load", lambda _: DummyModel())
    client = TestClient(app_main.app)

    payload = {
        "instances": [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        ]
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predictions"] == [1]
    assert response.json()["probabilities"] == [0.8]


def test_predict_endpoint_rejects_empty_payload(monkeypatch):
    monkeypatch.setattr(app_main, "MODEL", DummyModel(), raising=False)
    monkeypatch.setattr(app_main, "MODEL_PATHS", ["dummy-model-path.joblib"], raising=False)
    monkeypatch.setattr(app_main.os.path, "exists", lambda _: True)
    monkeypatch.setattr(app_main.joblib, "load", lambda _: DummyModel())
    client = TestClient(app_main.app)

    response = client.post("/predict", json={"instances": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "No instances provided"


def test_mlflow_summary_endpoint(monkeypatch):
    class DummyExperiment:
        experiment_id = "0"
        name = "Default"
        artifact_location = "/tmp/mlartifacts"
        lifecycle_stage = "active"

    class DummyRunInfo:
        run_id = "run-123"
        status = "FINISHED"
        start_time = 1234567890

    class DummyRunData:
        metrics = {"test_roc_auc": 0.91}
        params = {"candidate_model": "random_forest"}
        tags = {"mlflow.runName": "random_forest"}

    class DummyRun:
        info = DummyRunInfo()
        data = DummyRunData()

    class DummyMlflowClient:
        def search_experiments(self):
            return [DummyExperiment()]

        def search_runs(self, experiment_ids, order_by, max_results):
            assert experiment_ids == ["0"]
            assert max_results == 5
            return [DummyRun()]

    monkeypatch.setattr(app_main, "MlflowClient", lambda tracking_uri: DummyMlflowClient())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///tmp/mlflow.db")

    client = TestClient(app_main.app)
    response = client.get("/mlflow")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tracking_uri"] == "sqlite:///tmp/mlflow.db"
    assert payload["experiment_count"] == 1
    assert payload["experiments"][0]["name"] == "Default"
    assert payload["experiments"][0]["latest_runs"][0]["metrics"]["test_roc_auc"] == 0.91
