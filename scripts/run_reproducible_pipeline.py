"""Run the full local pipeline from a clean setup in a fixed order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step("Prepare data", [python, "scripts/download_and_prepare.py"])
    run_step("Generate EDA artifacts", [python, "eda/eda_generate.py"])
    run_step("Train and track models", [python, "mlflow/train_models.py"])
    run_step("Export final models", [python, "scripts/export_models.py"])
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
