import numpy as np
import pytest

from statstech.base import BaseEstimator, RegressorMixin
from statstech.model_selection.search import GridSearchCV, _expand_grid
from statstech.utils.validation import NotFittedError, check_is_fitted


# Auxiliares
# ----------
class ShiftRegressor(RegressorMixin, BaseEstimator):
    """Modelo de teste: prevê o próprio X somado a um deslocamento."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return X[:, 0] + self.shift


def make_data(n=20):
    """Devolve X (n, 1) e y (n,) com y igual a X."""
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()
    return X, y


# _expand_grid
# ------------
def test_expand_grid_cartesian_product():
    params = {"a": [1, 2], "b": [True, False]}
    combinations = _expand_grid(params)
    expected = [
        {"a": 1, "b": True},
        {"a": 1, "b": False},
        {"a": 2, "b": True},
        {"a": 2, "b": False},
    ]
    assert combinations == expected


# escolha do melhor
# -----------------
def test_best_params_picks_highest_r2():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)

    assert grid.best_params_ == {"shift": 0.0}
    assert grid.best_score_ == pytest.approx(1.0)


def test_lower_is_better_picks_smallest():
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, scoring="mse"
    ).fit(X, y)

    assert grid.best_params_ == {"shift": 0.0}
    assert grid.best_score_ == pytest.approx(0.0)


def test_best_score_matches_cv_results():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    scores = [
        r["mean_score"] for r in grid.cv_results_ if r["params"] == grid.best_params_
    ]
    assert len(scores) == 1
    assert scores[0] == pytest.approx(grid.best_score_)


# Resultados e entradas
# ---------------------
def test_cv_results_one_item_per_combination():
    """cv_results_ tem um item por combinação, na ordem da grade."""
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    assert len(grid.cv_results_) == 3
    for result in grid.cv_results_:
        assert set(result.keys()) == {"params", "mean_score"}
    params_list = [result["params"] for result in grid.cv_results_]
    assert params_list == [{"shift": -5.0}, {"shift": 0.0}, {"shift": 5.0}]


def test_original_estimator_not_modified():
    X, y = make_data()
    model = ShiftRegressor()
    GridSearchCV(model, {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert model.shift == 0.0


# refit, predict e score
# ----------------------
def test_refit_true_trains_best_estimator():
    X, y = make_data()
    model = ShiftRegressor()
    grid = GridSearchCV(model, {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    check_is_fitted(grid.best_estimator_)
    assert grid.best_estimator_.shift == grid.best_params_["shift"]


def test_predict_and_score_use_best_estimator():
    """predict e score delegam ao melhor modelo (shift=0)."""
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    np.testing.assert_allclose(grid.predict(X), X[:, 0])
    assert grid.score(X, y) == pytest.approx(1.0)


def test_refit_false_predict_raises():
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, refit=False
    ).fit(X, y)
    with pytest.raises(NotFittedError):
        grid.predict(X)
    assert grid.best_params_ == {"shift": 0.0}


def test_predict_before_fit_raises():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4)
    with pytest.raises(NotFittedError):
        grid.predict(X)
    with pytest.raises(NotFittedError):
        grid.score(X, y)
