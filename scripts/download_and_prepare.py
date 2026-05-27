"""Download/prepare local dataset and write cleaned CSV.
Reads `heart_disease/processed.cleveland.data`, replaces missing markers,
coerces types, maps target to binary, and writes `data/processed_cleveland.csv`.
"""
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "heart_disease", "processed.cleveland.data")
OUT_DIR = os.path.join(ROOT, "data")
OUT_CSV = os.path.join(OUT_DIR, "processed_cleveland.csv")

def prepare(in_path=RAW, out_path=OUT_CSV):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(in_path, header=None)
    # common UCI processed layout: 14 columns with target last
    if df.shape[1] == 14:
        cols = [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"
        ]
        df.columns = cols
    else:
        df = df.rename(columns={df.columns[-1]: "target"})

    df = df.replace("?", np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # binary target: 0 = no disease, 1 = disease (original labels >0)
    if df["target"].dtype.kind in "iuf":
        df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned CSV to {out_path}")

if __name__ == "__main__":
    prepare()
