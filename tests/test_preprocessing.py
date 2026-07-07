import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor


def test_build_preprocessor_transforms_numeric_and_categorical_features():
    df = pd.DataFrame(
        {
            "age": [63, 41, np.nan],
            "chol": [233, np.nan, 204],
            "cp_label": ["typical", "atypical", None],
            "target": [0, 1, 0],
        }
    )

    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))

    assert transformed.shape[0] == len(df)
    assert np.isfinite(transformed).all()


def test_build_preprocessor_excludes_target_from_features():
    df = pd.DataFrame(
        {
            "feature_num": [1.0, 2.0, 3.0],
            "feature_cat": ["a", "b", "a"],
            "target": [0, 1, 0],
        }
    )

    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))

    assert transformed.shape[0] == 3
