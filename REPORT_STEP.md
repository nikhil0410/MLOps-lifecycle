# MLOps Assignment 01 Report

**Course:** Machine Learning Operations (MLOps) AIMLCZG523  
**Assignment:** Assignment 01  
**Project:** Heart Disease Risk Prediction - End-to-End MLOps Pipeline  
**Student Name:** `<Your Name>`  
**Student ID:** `<Your ID>`  
**Repository Link:** `<Add GitHub Repository URL>`  
**Deployment Type:** Docker Desktop Kubernetes / Local Kubernetes  
**Deployed API Access:** `<Add local access instructions or public URL>`

---

## 1. Executive Summary

This project implements an end-to-end MLOps workflow for predicting heart disease risk using the UCI Heart Disease dataset. The solution covers data preparation, exploratory data analysis, feature engineering, model training, experiment tracking, model packaging, CI pipeline automation, containerization, Kubernetes deployment, and application monitoring.

The final application is exposed as a FastAPI service with:

- `GET /health`
- `POST /predict`
- `GET /metrics`

The deployment was performed on a local Kubernetes cluster using Docker Desktop Kubernetes.

---

## 2. Problem Statement

The objective is to build a binary classification system that predicts the presence or absence of heart disease from patient clinical attributes such as age, sex, blood pressure, cholesterol, and related health indicators.

The project also aims to demonstrate production-oriented MLOps practices including:

- reproducible pipelines
- experiment tracking
- CI automation
- Docker packaging
- Kubernetes deployment
- monitoring and logging

---

## 3. Dataset Description

**Dataset:** UCI Heart Disease Dataset  
**Source:** UCI Machine Learning Repository

The dataset contains patient-level medical information and a target variable indicating whether heart disease is present.

### Main features

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal
- target

### Data preparation summary

- missing values represented as `?` were converted to null values
- target values were converted into binary labels
- processed output was written as a cleaned CSV for downstream tasks

---

## 4. Project Structure

Example project layout:

```text
MLOps-lifecycle/
??? app/
??? artifacts/
??? data/
??? eda/
??? heart_disease/
??? k8s/
??? mlflow/
??? models/
??? scripts/
??? src/
??? tests/
??? .github/workflows/
??? Dockerfile
??? requirements.txt
??? requirements-dev.txt
??? requirements-prod.txt
??? README.md
```

---

## 5. Setup and Installation

### Local Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Development dependencies

```bash
python3 -m pip install -r requirements-dev.txt
```

### Conda setup

```bash
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## 6. Data Acquisition and EDA

The data preparation script cleans the source dataset and generates a processed CSV used across training and export scripts.

### EDA performed

- feature distribution plots
- correlation heatmap
- class balance visualization

### EDA observations

Add your own observations here, for example:

- whether the target classes are balanced
- which numerical features show visible spread
- whether any features seem correlated with the target

### Evidence

Insert screenshots from:

- `artifacts/eda/`
- notebook outputs if required

---

## 7. Feature Engineering and Model Development

### Preprocessing

The project uses a reusable preprocessing pipeline that:

- imputes missing numerical values
- scales numerical values
- imputes categorical values
- one-hot encodes categorical values

### Candidate models

- Logistic Regression
- Random Forest

### Model selection process

Model tuning and comparison were performed using `GridSearchCV`.

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Selected model

Add the selected model here after reviewing:

- `artifacts/models/model_selection.json`
- `artifacts/models/model_selection_summary.md`
- `artifacts/models/leaderboard.csv`

### Why the model was chosen

Add a short explanation here, such as:

- best ROC-AUC
- stable performance across folds
- acceptable tradeoff between recall and precision

---

## 8. Experiment Tracking

MLflow was used to track model experiments.

Tracked items include:

- tuned parameters
- evaluation metrics
- prediction artifacts
- confusion matrices
- ROC curves
- saved model artifacts

### Tracking backend

The project uses a local SQLite MLflow tracking backend:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### MLflow UI

```bash
mlflow ui
```

### Evidence

Insert screenshots of:

- experiment runs
- metrics
- artifacts
- registered logged models if visible

---

## 9. Model Packaging and Reproducibility

The final model is packaged in reusable serialized form and includes the preprocessing pipeline to ensure reproducibility.

### Reproducibility assets

- `requirements.txt`
- `requirements-dev.txt`
- `requirements-prod.txt`
- `environment.yml`
- `scripts/run_reproducible_pipeline.py`
- exported model pipelines in `models/`

### Reproducible pipeline execution

```bash
python3 scripts/run_reproducible_pipeline.py
```

### Exported models

- `models/best_model_pipeline.joblib`
- `models/random_forest_pipeline.joblib`
- `models/logistic_pipeline.joblib`

---

## 10. CI/CD Pipeline

GitHub Actions was configured as a multi-job CI pipeline.

### Pipeline jobs

- `lint`
- `unit-tests`
- `train-and-export`

### Pipeline behavior

- lint checks run first
- unit tests run independently
- model training and export run only after lint and tests pass

### Automated checks performed

- Ruff linting
- Pytest unit tests
- data preparation
- model training
- model export

### Evidence

Insert screenshots of:

- CI workflow summary
- passing jobs
- uploaded artifacts

---

## 11. Containerization

The application is containerized using Docker.

### Build command

```bash
docker build -t heart-disease-api:v2 .
```

### Run locally

```bash
docker run -p 8000:8000 --rm heart-disease-api:v2
```

### API endpoints

- `GET /health`
- `POST /predict`
- `GET /metrics`

### Evidence

Insert screenshots of:

- Docker image in Docker Desktop
- local container run
- API response

---

## 12. Kubernetes Deployment

The application was deployed to local Kubernetes using Docker Desktop Kubernetes.

### Kubernetes resources

- Deployment
- Service

Files used:

- `k8s/deployment.yaml`
- `k8s/service.yaml`

### Deployment commands

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Access method

```bash
kubectl port-forward svc/heart-disease-api-service 8000:80
```

### Verification

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/predict
```

### Evidence

Insert screenshots of:

- `kubectl get pods`
- `kubectl get svc`
- `/health`
- `/predict`

---

## 13. Monitoring and Logging

The deployed API includes lightweight monitoring and logging support.

### Logging

The application logs:

- request method
- request path
- status code
- request duration
- prediction batch size
- prediction failures

### Metrics

The `/metrics` endpoint exposes Prometheus-style metrics such as:

- `heart_disease_api_requests_total`
- `heart_disease_api_request_latency_seconds`
- `heart_disease_api_predictions_total`

### Kubernetes annotations

Prometheus scrape annotations are attached to the Kubernetes deployment for simple metrics collection readiness.

### Evidence

Insert screenshots of:

- `/metrics`
- `kubectl logs <pod-name>`
- optional pod annotations

---

## 14. Architecture Diagram

Add a simple architecture diagram showing:

1. dataset input
2. preprocessing and training pipeline
3. MLflow tracking
4. exported model artifacts
5. FastAPI application
6. Docker image
7. Kubernetes deployment
8. monitoring via `/metrics`

A Mermaid version of the architecture is available in:

- `ARCHITECTURE_DIAGRAM.md`

You can create a rendered image from that Mermaid diagram using:

- draw.io
- PowerPoint
- Excalidraw
- Lucidchart
- mermaid.live

Insert the final image in the report here.

---

## 15. Challenges and Resolutions

Document the major issues faced during development. Example items:

- CI dependency issues
- MLflow backend configuration issues
- MLflow model serialization issues
- Kubernetes image pull issues
- service routing or port-forward issues
- metrics endpoint validation

For each issue, describe:

- what happened
- why it happened
- how it was resolved

---

## 16. Conclusion

This project successfully demonstrates an end-to-end MLOps workflow for a heart disease prediction problem. The solution includes data preparation, model development, experiment tracking, CI automation, containerization, local Kubernetes deployment, and application monitoring. The final result is a reproducible and deployable ML system aligned with the practical objectives of the assignment.

---

## 17. Submission Checklist

Before final submission, confirm the following:

- [ ] repository link added
- [ ] student details added
- [ ] screenshots inserted
- [ ] deployment proof added
- [ ] CI proof added
- [ ] MLflow proof added
- [ ] architecture diagram inserted
- [ ] local access instructions included
- [ ] final report exported as PDF/doc/docx
- [ ] short demo video recorded
