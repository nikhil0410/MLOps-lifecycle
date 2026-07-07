from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling import RANDOM_STATE, TARGET_COLUMN, build_model_candidates


def test_model_candidates_include_expected_estimators_and_grids():
    candidates = build_model_candidates()

    assert TARGET_COLUMN == "target"
    assert set(candidates) == {"logistic_regression", "random_forest"}
    assert isinstance(candidates["logistic_regression"]["estimator"], LogisticRegression)
    assert isinstance(candidates["random_forest"]["estimator"], RandomForestClassifier)
    assert candidates["logistic_regression"]["estimator"].random_state == RANDOM_STATE
    assert "clf__C" in candidates["logistic_regression"]["param_grid"]
    assert "clf__n_estimators" in candidates["random_forest"]["param_grid"]
