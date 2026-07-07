"""Export tuned pipelines and selected-model metadata for serving."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.modeling import TARGET_COLUMN, build_model_candidates
from src.preprocessing import build_preprocessor

ROOT = Path(ROOT_DIR)
CSV = ROOT / "data" / "processed_cleveland.csv"
OUT_DIR = ROOT / "models"
SELECTION_JSON = ROOT / "artifacts" / "models" / "model_selection.json"
METADATA_JSON = OUT_DIR / "model_metadata.json"


def ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(CSV)
    X = df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df.iloc[:, :-1]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else df.iloc[:, -1]
    return df, X, y


def load_selection() -> dict:
    if not SELECTION_JSON.exists():
        return {}
    return json.loads(SELECTION_JSON.read_text())


def main() -> None:
    ensure()
    selection = load_selection()
    df, X, y = load_data()
    pre = build_preprocessor(df)
    legacy_aliases = {
        "logistic_regression": "logistic_pipeline.joblib",
    }

    export_metadata = {
        "selected_model": selection.get("selected_model"),
        "selection_metric": selection.get("selection_metric"),
        "exported_models": {},
    }

    for name, config in build_model_candidates().items():
        clf = config["estimator"]
        if selection.get("selected_model") == name:
            clf.set_params(**{k.replace("clf__", ""): v for k, v in selection["selected_params"].items()})

        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X, y)

        out_path = OUT_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, out_path)
        export_metadata["exported_models"][name] = {
            "path": out_path.name,
            "params": pipe.get_params(),
        }
        print(f"Saved {out_path}")

        legacy_name = legacy_aliases.get(name)
        if legacy_name:
            legacy_path = OUT_DIR / legacy_name
            joblib.dump(pipe, legacy_path)
            export_metadata["exported_models"][name]["legacy_path"] = legacy_path.name
            print(f"Saved {legacy_path}")

        if selection.get("selected_model") == name:
            best_path = OUT_DIR / "best_model_pipeline.joblib"
            joblib.dump(pipe, best_path)
            export_metadata["best_model_path"] = best_path.name
            print(f"Saved {best_path}")

    if "best_model_path" not in export_metadata:
        default_best = OUT_DIR / "random_forest_pipeline.joblib"
        export_metadata["best_model_path"] = default_best.name
        export_metadata["selected_model"] = export_metadata["selected_model"] or "random_forest"

    METADATA_JSON.write_text(json.dumps(export_metadata, indent=2, default=str))
    print(f"Saved {METADATA_JSON}")


if __name__ == "__main__":
    main()
