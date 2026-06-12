MLflow training helper

Run the training script to log a RandomForest model and a sample predictions artifact.

Install dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 mlflow/train_mlflow.py
```

This will create an `mlruns/` directory containing the logged run. You can view it locally with:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in your browser.
