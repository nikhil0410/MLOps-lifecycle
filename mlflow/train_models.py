"""Train, tune, compare, and track candidate models with MLflow."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "mlflow is required to run this script. Install project dependencies with "
        "`python3 -m pip install -r requirements.txt` first."
    ) from exc
from mlflow.models import infer_signature

# ensure project root is on sys.path when running this script from the mlflow/ folder
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling import RANDOM_STATE, TARGET_COLUMN, build_model_candidates  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

ROOT = ROOT_DIR
CSV = ROOT / "data" / "processed_cleveland.csv"
OUT = ROOT / "artifacts" / "models"
SELECTION_JSON = OUT / "model_selection.json"
LEADERBOARD_CSV = OUT / "leaderboard.csv"
SUMMARY_MD = OUT / "model_selection_summary.md"


def ensure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV)


def build_search(name: str, estimator, param_grid: dict, train_df: pd.DataFrame) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("pre", build_preprocessor(train_df)),
            ("clf", estimator),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        verbose=0,
    )


def save_cv_results(name: str, search: GridSearchCV) -> Path:
    df = pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc")
    keep_cols = [
        "rank_test_roc_auc",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "params",
    ]
    out_path = OUT / f"{name}_cv_results.csv"
    df[keep_cols].to_csv(out_path, index=False)
    return out_path


def evaluate_holdout(name: str, estimator, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, Path, Path, Path]:
    preds = estimator.predict(X_test)
    probs = estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds),
        "test_recall": recall_score(y_test, preds),
        "test_f1": f1_score(y_test, preds),
        "test_roc_auc": roc_auc_score(y_test, probs),
    }

    pred_path = OUT / f"{name}_holdout_predictions.csv"
    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": preds,
            "y_proba": probs,
        }
    ).to_csv(pred_path, index=False)

    cm_path = OUT / f"{name}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, colorbar=False)
    ax.set_title(f"{name} confusion matrix")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    roc_path = OUT / f"{name}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probs, ax=ax)
    ax.set_title(f"{name} ROC curve")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    return metrics, pred_path, cm_path, roc_path


def write_selection_artifacts(leaderboard: pd.DataFrame) -> None:
    leaderboard.to_csv(LEADERBOARD_CSV, index=False)
    best_row = leaderboard.sort_values(
        ["best_cv_roc_auc", "test_roc_auc"],
        ascending=False,
    ).iloc[0]
    selection_payload = {
        "selection_metric": "best_cv_roc_auc",
        "selected_model": best_row["model_name"],
        "selected_params": json.loads(best_row["best_params_json"]),
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    SELECTION_JSON.write_text(json.dumps(selection_payload, indent=2))

    lines = [
        "# Model Selection Summary",
        "",
        f"- Selection metric: `{selection_payload['selection_metric']}`",
        f"- Selected model: `{selection_payload['selected_model']}`",
        f"- Selected model CV ROC-AUC: `{best_row['best_cv_roc_auc']:.4f}`",
        f"- Selected model test ROC-AUC: `{best_row['test_roc_auc']:.4f}`",
        "",
        "## Candidate comparison",
        "",
        "| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC | Test ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            "| "
            f"{row['model_name']} | "
            f"{row['best_cv_accuracy']:.4f} | "
            f"{row['best_cv_precision']:.4f} | "
            f"{row['best_cv_recall']:.4f} | "
            f"{row['best_cv_f1']:.4f} | "
            f"{row['best_cv_roc_auc']:.4f} | "
            f"{row['test_roc_auc']:.4f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure()

    df = load_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_df = pd.concat([X_train, y_train], axis=1)

    leaderboard_rows = []
    for name, config in build_model_candidates().items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("candidate_model", name)
            search = build_search(name, config["estimator"], config["param_grid"], train_df)
            search.fit(X_train, y_train)

            cv_results_path = save_cv_results(name, search)
            mlflow.log_artifact(str(cv_results_path))

            holdout_metrics, pred_path, cm_path, roc_path = evaluate_holdout(
                name,
                search.best_estimator_,
                X_test,
                y_test,
            )
            mlflow.log_metrics(holdout_metrics)
            mlflow.log_artifact(str(pred_path))
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_dict(search.best_params_, f"{name}_best_params.json")
            mlflow.log_params(
                {
                    f"best_{param_name.replace('clf__', '')}": value
                    for param_name, value in search.best_params_.items()
                }
            )
            input_example = X_train.head(5)
            signature = infer_signature(input_example, search.best_estimator_.predict(input_example))
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name=f"model_{name}",
                input_example=input_example,
                signature=signature,
                serialization_format="cloudpickle",
            )

            leaderboard_rows.append(
                {
                    "model_name": name,
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                    "best_cv_accuracy": search.cv_results_["mean_test_accuracy"][search.best_index_],
                    "best_cv_precision": search.cv_results_["mean_test_precision"][search.best_index_],
                    "best_cv_recall": search.cv_results_["mean_test_recall"][search.best_index_],
                    "best_cv_f1": search.cv_results_["mean_test_f1"][search.best_index_],
                    "best_cv_roc_auc": search.cv_results_["mean_test_roc_auc"][search.best_index_],
                    **holdout_metrics,
                }
            )
            print(f"Completed tuned run for {name}.")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["best_cv_roc_auc", "test_roc_auc"],
        ascending=False,
    )
    write_selection_artifacts(leaderboard)
    print(f"Saved leaderboard to {LEADERBOARD_CSV}")
    print(f"Saved selection summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
