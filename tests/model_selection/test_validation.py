import numpy as np
import pytest

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statstech.metrics.classification import log_loss
from statstech.metrics.regression import mean_squared_error
from statstech.model_selection.split import KFold, LeaveOneOut
from statstech.model_selection.validation import cross_val_score
from statstech.utils.validation import NotFittedError, check_is_fitted


# Auxiliares: modelos, scorings e dados de teste
# ----------------------------------------------
class MeanRegressor(RegressorMixin, BaseEstimator):
    """Modelo de teste: prevê sempre a média do y de treino."""

    def fit(self, X, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class ConstantScoreRegressor(MeanRegressor):
    """Modelo de teste cujo score é sempre 7.0"""

    def score(self, X, y):
        return 7.0


class ZeroClassifier(ClassifierMixin, BaseEstimator):
    """Modelo de teste: prevê sempre a classe 0 (a menor).

    As probabilidades são fixas: 0.8 para a classe 0 e 0.2 para a outra.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        proba = np.full((len(X), 2), 0.2)
        proba[:, 0] = 0.8
        return proba


def constant_scoring(y_true, y_pred):
    return 10.0


def amp_pred_scoring(y_true, y_pred):
    return np.ptp(y_pred)


def make_data(n=20):
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n) * 2.0
    return X, y


def make_class_data(n=20):
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n) % 2
    return X, y


# cross_val_score: básico
# -----------------------
def test_one_score_per_fold():
    X, y = make_data()
    scores = cross_val_score(MeanRegressor(), X, y, cv=4)
    assert scores.shape == (4,)


def test_original_estimator_not_fitted():
    X, y = make_data()
    model = MeanRegressor()
    cross_val_score(model, X, y, cv=4)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


# cross_vall_score: scoring
# -------------------------
def test_default_scoring_uses_estimator_score():
    X, y = make_data()
    scores = cross_val_score(ConstantScoreRegressor(), X, y, cv=4)
    np.testing.assert_allclose(scores, 7.0)


def test_given_scoring_is_used():
    X, y = make_data()
    scores = cross_val_score(
        ConstantScoreRegressor(), X, y, cv=4, scoring=constant_scoring
    )
    np.testing.assert_allclose(scores, 10.0)


def test_str_scoring_is_used():
    X, y = make_class_data()
    scores = cross_val_score(ZeroClassifier(), X, y, cv=4, scoring="recall_macro")
    np.testing.assert_allclose(scores, 0.5)


def test_needs_proba_scoring_is_used():
    X, y = make_class_data()
    scores = cross_val_score(ZeroClassifier(), X, y, cv=4, scoring="log_loss")
    proba = np.tile([0.8, 0.2], (5, 1))
    fold_a = log_loss(np.array([0, 1, 0, 1, 0]), proba, [0, 1])
    fold_b = log_loss(np.array([1, 0, 1, 0, 1]), proba, [0, 1])
    expected = np.array([fold_a, fold_b, fold_a, fold_b])
    np.testing.assert_allclose(scores, expected)


def test_given_scoring_right_order():
    X, y = make_data()
    scores = cross_val_score(MeanRegressor(), X, y, cv=4, scoring=amp_pred_scoring)
    np.testing.assert_allclose(scores, 0.0)


def test_str_equals_function():
    X, y = make_data()
    scores_str_mse = cross_val_score(MeanRegressor(), X, y, cv=4, scoring="mse")
    scores_func_mse = cross_val_score(
        MeanRegressor(), X, y, cv=4, scoring=mean_squared_error
    )
    np.testing.assert_allclose(scores_str_mse, scores_func_mse)


# cross_val_score: parâmetro cv
# -----------------------------
def test_int_cv_kfold():
    X, y = make_data()
    scores_int = cross_val_score(MeanRegressor(), X, y, cv=4)
    scores_kfold = cross_val_score(MeanRegressor(), X, y, cv=KFold(n_splits=4))
    np.testing.assert_allclose(scores_int, scores_kfold)


def test_cv_object_is_used():
    X, y = make_data()
    scores_default = cross_val_score(MeanRegressor(), X, y, cv=KFold(n_splits=4))
    scores_shuffled = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    assert not np.allclose(scores_default, scores_shuffled)


def test_cv_same_seed_same_scores():
    X, y = make_data()
    scores_1 = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    scores_2 = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    np.testing.assert_allclose(scores_1, scores_2)


def test_cv_accepts_loo():
    X, y = make_data()
    n = len(y)
    scores_loo = cross_val_score(ConstantScoreRegressor(), X, y, cv=LeaveOneOut())
    assert scores_loo.shape == (n,)
    np.testing.assert_allclose(scores_loo, 7.0)


# cross_val_score: entradas inválidas
# -----------------------------------
def test_int_scoring_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="scoring deve ser"):
        cross_val_score(MeanRegressor(), X, y, scoring=123)


def test_invalid_str_scoring_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="scoring desconhecido"):
        cross_val_score(MeanRegressor(), X, y, scoring="xyz")


def test_X_1d_raises():
    X, y = np.arange(20), np.arange(20)
    with pytest.raises(ValueError, match="precisa ser 2D"):
        cross_val_score(MeanRegressor(), X, y, cv=4)


def test_X_y_diff_lengths_raises():
    X, _ = make_data()
    y = np.arange(18)
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        cross_val_score(MeanRegressor(), X, y, cv=4)


def test_needs_proba_scoring_no_predict_proba_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="predict_proba"):
        cross_val_score(MeanRegressor(), X, y, scoring="log_loss")
