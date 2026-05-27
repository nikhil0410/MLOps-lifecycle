# Heart Disease MLOps Project

Project layout and quick commands.

- `data/` — processed dataset (generated)
- `src/` — preprocessing and helpers
- `scripts/` — data prep and export helpers
- `mlflow/` — training scripts and MLflow examples
- `artifacts/` — EDA and CV artifacts
- `models/` — exported model pipelines (joblib)
- `notebooks/` — notebooks (move existing ones here)

Quick commands:

```bash
# Prepare data
python3 scripts/download_and_prepare.py

# Generate EDA artifacts
python3 eda/eda_generate.py

# Train and log models (MLflow)
python3 mlflow/train_models.py

# Export final models to models/
python3 scripts/export_models.py
```

View MLflow UI:

```bash
mlflow ui
# then open http://127.0.0.1:5000
```
