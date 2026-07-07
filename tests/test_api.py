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
