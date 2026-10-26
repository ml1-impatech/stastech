import numpy as np
import pytest

from statstech.base import clone
from statstech.linear_model import LogisticRegression, Ridge
from statstech.metrics.regression import r2_score
from statstech.model_selection import GridSearchCV, KFold, cross_val_score
from statstech.pipeline import Pipeline
from statstech.preprocessing import StandardScaler
from statstech.preprocessing.polynomial import PolynomialFeatures
from statstech.utils.validation import NotFittedError, check_is_fitted


# Auxiliares
# ----------
def make_regression_data(n=60):
    """X com colunas em escalas bem diferentes e y linear em X."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3)) * [1.0, 100.0, 0.01]
    y = X[:, 0] + X[:, 1] / 100 + rng.normal(scale=0.1, size=n)
    return X, y


def make_classification_data(n=60):
    """X com duas colunas e y com duas classes."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 2)) * [1.0, 50.0]
    y = (X[:, 0] + X[:, 1] / 50 > 0).astype(int)
    return X, y


def make_pipeline(model=None):
    return Pipeline([("scaler", StandardScaler()), ("model", model or Ridge())])


# fit e previsões
# ---------------
def test_fit_returns_self():
    X, y = make_regression_data()
    pipe = make_pipeline()
    assert pipe.fit(X, y) is pipe


def test_matches_manual_steps():
    X, y = make_regression_data()
    pipe = make_pipeline().fit(X, y)

    scaler = StandardScaler().fit(X)
    model = Ridge().fit(scaler.transform(X), y)
    np.testing.assert_allclose(pipe.predict(X), model.predict(scaler.transform(X)))


def test_scaler_learns_only_fit_data():
    X, y = make_regression_data()
    pipe = make_pipeline().fit(X[:30], y[:30])
    np.testing.assert_allclose(pipe.named_steps_["scaler"].mean_, X[:30].mean(axis=0))


def test_original_steps_are_not_fitted():
    X, y = make_regression_data()
    scaler, model = StandardScaler(), Ridge()
    pipe = Pipeline([("scaler", scaler), ("model", model)]).fit(X, y)

    assert pipe.steps_[0][1] is not scaler
    assert pipe.steps_[1][1] is not model
    with pytest.raises(NotFittedError):
        check_is_fitted(scaler)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_named_steps_point_to_fitted_steps():
    X, y = make_regression_data()
    pipe = make_pipeline().fit(X, y)
    assert pipe.named_steps_["model"] is pipe.steps_[1][1]
    assert pipe.named_steps_["model"].coef_.shape == (3,)


def test_three_steps():
    X, y = make_regression_data()
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    ).fit(X, y)
    assert pipe.named_steps_["model"].coef_.shape == (9,)
    assert pipe.predict(X).shape == (len(y),)


# score, predict_proba e classes_
# -------------------------------
def test_score_of_regressor_is_r2():
    X, y = make_regression_data()
    pipe = make_pipeline().fit(X, y)
    assert pipe.score(X, y) == pytest.approx(r2_score(y, pipe.predict(X)))


def test_score_of_classifier_is_accuracy():
    X, y = make_classification_data()
    pipe = make_pipeline(LogisticRegression()).fit(X, y)
    assert pipe.score(X, y) == pytest.approx(np.mean(pipe.predict(X) == y))


def test_predict_proba_and_classes():
    X, y = make_classification_data()
    pipe = make_pipeline(LogisticRegression()).fit(X, y)
    proba = pipe.predict_proba(X)
    model = pipe.named_steps_["model"]
    scaled = pipe.named_steps_["scaler"].transform(X)

    np.testing.assert_array_equal(pipe.classes_, [0, 1])
    np.testing.assert_allclose(proba, model.predict_proba(scaled))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_regressor_has_no_classes_nor_predict_proba():
    X, y = make_regression_data()
    pipe = make_pipeline().fit(X, y)
    assert not hasattr(pipe, "classes_")
    with pytest.raises(AttributeError):
        pipe.predict_proba(X)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "score"])
def test_methods_before_fit_raise(method):
    X, y = make_classification_data()
    pipe = make_pipeline(LogisticRegression())
    args = (X, y) if method == "score" else (X,)
    with pytest.raises(NotFittedError):
        getattr(pipe, method)(*args)


# set_params e clone
# ------------------
def test_set_params_changes_step_param():
    pipe = make_pipeline()
    pipe.set_params(model__alpha=5.0)
    assert pipe.steps[1][1].alpha == 5.0


def test_set_params_own_param():
    pipe = make_pipeline()
    new_steps = [("model", Ridge(alpha=3.0))]
    pipe.set_params(steps=new_steps)
    assert pipe.steps is new_steps


def test_set_params_unknown_step_raises():
    with pytest.raises(ValueError, match="não existe"):
        make_pipeline().set_params(modelo__alpha=1.0)


def test_set_params_unknown_param_of_step_raises():
    with pytest.raises(ValueError):
        make_pipeline().set_params(model__beta=1.0)


def test_clone_copies_steps():
    pipe = make_pipeline(Ridge(alpha=2.0))
    copy = clone(pipe)
    assert copy is not pipe
    assert copy.steps[1][1] is not pipe.steps[1][1]
    assert copy.steps[1][1].alpha == 2.0


# validação cruzada e busca
# -------------------------
def test_cross_val_score_refits_scaler_in_each_fold():
    X, y = make_regression_data()
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    scores = cross_val_score(make_pipeline(), X, y, cv=cv)

    expected = []
    for train_idx, test_idx in cv.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        model = Ridge().fit(scaler.transform(X[train_idx]), y[train_idx])
        expected.append(model.score(scaler.transform(X[test_idx]), y[test_idx]))
    np.testing.assert_allclose(scores, expected)


def test_grid_search_with_step_params():
    X, y = make_regression_data()
    pipe = make_pipeline()
    grid = GridSearchCV(
        pipe, {"model__alpha": [0.01, 1.0, 1000.0]}, cv=4, scoring="rmse"
    ).fit(X, y)

    assert grid.best_params_ in [{"model__alpha": a} for a in [0.01, 1.0]]
    assert (
        grid.best_estimator_.named_steps_["model"].alpha
        == (grid.best_params_["model__alpha"])
    )
    assert pipe.steps[1][1].alpha == 1.0


def test_grid_search_without_scoring_uses_score():
    X, y = make_regression_data()
    grid = GridSearchCV(make_pipeline(), {"model__alpha": [0.1, 10.0]}, cv=4)
    grid.fit(X, y)
    assert 0 < grid.best_score_ <= 1


# steps inválidos
# ---------------
class NoTransform:
    """Objeto de teste com fit, mas sem transform."""

    def fit(self, X, y=None):
        return self


@pytest.mark.parametrize(
    "steps",
    [
        [],
        "scaler",
        None,
        [StandardScaler()],
        [("scaler", StandardScaler(), "extra")],
        [(1, Ridge())],
        [("a__b", Ridge())],
        [("model", StandardScaler()), ("model", Ridge())],
        [("step", NoTransform()), ("model", Ridge())],
        [("scaler", StandardScaler()), ("model", object())],
    ],
)
def test_invalid_steps_raise(steps):
    X, y = make_regression_data()
    with pytest.raises(ValueError):
        Pipeline(steps).fit(X, y)
