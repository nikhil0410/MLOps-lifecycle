"""Reusable preprocessing pipeline builder.
Provides a ColumnTransformer that imputes, scales numeric features and one-hot encodes categoricals.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def build_preprocessor(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # exclude target if present
    if "target" in num_cols:
        num_cols.remove("target")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # OneHotEncoder parameter name differs between sklearn versions (sparse vs sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre
