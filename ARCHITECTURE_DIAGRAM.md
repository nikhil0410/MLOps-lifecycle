# Architecture Diagram

The following Mermaid diagram represents the end-to-end MLOps architecture used in this project.

```mermaid
flowchart TD
    A[UCI Heart Disease Dataset] --> B[Data Preparation Script<br/>scripts/download_and_prepare.py]
    B --> C[Processed Dataset<br/>data/processed_cleveland.csv]

    C --> D[EDA Pipeline<br/>eda/eda_generate.py]
    D --> E[EDA Artifacts<br/>histograms, heatmap, class balance]

    C --> F[Preprocessing Pipeline<br/>src/preprocessing.py]
    F --> G[Model Training and Tuning<br/>mlflow/train_models.py]
    G --> H[Candidate Models<br/>Logistic Regression<br/>Random Forest]
    G --> I[Cross Validation and Metrics<br/>Accuracy, Precision, Recall, F1, ROC-AUC]

    G --> J[MLflow Tracking]
    J --> K[Parameters, Metrics, Artifacts, Logged Models]

    G --> L[Model Selection Artifacts<br/>leaderboard.csv, model_selection.json]
    G --> M[Model Export<br/>scripts/export_models.py]
    M --> N[Exported Model Pipelines<br/>models/*.joblib]

    N --> O[FastAPI Application<br/>app/main.py]
    O --> P[REST API Endpoints<br/>/health, /predict, /metrics]

    P --> Q[Docker Image<br/>Dockerfile]
    Q --> R[Kubernetes Deployment<br/>k8s/deployment.yaml]
    R --> S[Kubernetes Service<br/>k8s/service.yaml]

    S --> T[Local Kubernetes Access<br/>kubectl port-forward]
    T --> U[User / Evaluator]

    O --> V[Application Logging]
    O --> W[Prometheus Metrics Endpoint<br/>/metrics]
    W --> X[Monitoring / Scraping Ready<br/>Prometheus-compatible]

    Y[GitHub Actions CI Pipeline] --> Z[Lint + Unit Tests + Train and Export]
    Z --> J
    Z --> N
```

## Components Covered

This architecture includes the assignment-required components:

- dataset
- preprocessing and training
- MLflow
- model export
- FastAPI
- Docker
- Kubernetes
- monitoring

## How to Use It in the Report

You can:

1. copy the Mermaid block directly into your markdown report if the renderer supports Mermaid
2. render it using a Mermaid editor and save it as an image
3. place the rendered image in `screenshots/12_architecture_diagram.png`

If you want a rendered version, you can paste the Mermaid code into:

- https://mermaid.live/

and export it as PNG or SVG for the final report.
