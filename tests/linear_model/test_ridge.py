import numpy as np
import pytest

from statstech.linear_model.linear_regression import LinearRegression
from statstech.linear_model.ridge import Ridge


def test_ridge_alpha_zero_matches_linear_LinearRegression():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 2]], dtype=float)

    y = np.array([4, 7, 8, 11, 16], dtype=float)

    model_ridge = Ridge(alpha=0.0)
    model_ridge.fit(X, y)

    model_linear = LinearRegression()
    model_linear.fit(X, y)

    np.testing.assert_allclose(model_ridge.coef_, model_linear.coef_, 1e-8)
    np.testing.assert_allclose(model_ridge.intercept_, model_linear.intercept_, 1e-8)


def test_ridge_coefficients_norm_decrease_as_alpha_increases():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 2]], dtype=float)

    y = np.array([4, 7, 8, 11, 16])

    model_alpha_small = Ridge(alpha=0.1)
    model_alpha_medium = Ridge(alpha=10)
    model_alpha_large = Ridge(alpha=100.0)

    model_alpha_small.fit(X, y)
    model_alpha_medium.fit(X, y)
    model_alpha_large.fit(X, y)

    norm_alpha_small = np.linalg.norm(model_alpha_small.coef_)
    norm_alpha_medium = np.linalg.norm(model_alpha_medium.coef_)
    norm_alpha_large = np.linalg.norm(model_alpha_large.coef_)

    assert norm_alpha_small > norm_alpha_medium
    assert norm_alpha_medium > norm_alpha_large


def test_ridge_large_alpha_coefficients_zero_and_intercept_mean_y():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 2]], dtype=float)

    y = np.array([4, 7, 8, 11, 16], dtype=float)

    model = Ridge(alpha=1e10)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, np.zeros(X.shape[1]), atol=1e-6)
    np.testing.assert_allclose(model.intercept_, np.mean(y), atol=1e-6)


def test_ridge_default_parameters():
    model = Ridge()

    assert model.alpha == 1.0
    assert model.fit_intercept is True


def test_ridge_invalid_alpha_raises_value_error():
    X = np.array([[1, 3], [4, 7], [5, 6]], dtype=float)

    y = np.array([3, 5, 7], dtype=float)

    model = Ridge(alpha=-0.1)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ridge_invalid_fit_intercept_raises_value_error():
    X = np.array([[1, 3], [4, 7], [5, 6]], dtype=float)

    y = np.array([3, 5, 7], dtype=float)

    model = Ridge(fit_intercept=1)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ridge_fit_returns_self():
    X = np.array([[1, 3], [4, 7], [5, 6]], dtype=float)

    y = np.array([3, 5, 7], dtype=float)

    model = Ridge()

    assert model.fit(X, y) is model


def test_ridge_fit_multiple_features():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]], dtype=float)

    y = np.array([4, 5, 10, 11], dtype=float)

    model = Ridge()
    model.fit(X, y)

    assert model.coef_.shape == (2,)
    assert model.n_features_in_ == 2
    assert np.isfinite(model.coef_).all()


def test_ridge_fit_without_intercept():
    X = np.array([[1], [2], [3]], dtype=float)

    y = np.array([2, 4, 6], dtype=float)

    model = Ridge(fit_intercept=False)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, [28 / 15])
    assert model.intercept_ == 0.0


def test_ridge_fit_inconsistent_samples():
    X = np.array([[1], [2], [3]], dtype=float)

    y = np.array([3, 5], dtype=float)

    model = Ridge()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ridge_predict():
    X = np.array([[1], [2], [3]], dtype=float)

    y = np.array([3, 5, 7], dtype=float)

    model = Ridge(alpha=0.0)
    model.fit(X, y)

    X_test = np.array([[4], [5]], dtype=float)

    pred = model.predict(X_test)

    np.testing.assert_allclose(pred, [9, 11])


def test_ridge_predict_without_fit():
    model = Ridge()
    X = np.array([[1], [2]], dtype=float)

    with pytest.raises(ValueError):
        model.predict(X)


def test_ridge_predict_wrong_number_of_features():
    X_train = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)

    y_train = np.array([5, 8, 11], dtype=float)

    model = Ridge()
    model.fit(X_train, y_train)

    X_test = np.array([[7]], dtype=float)

    with pytest.raises(ValueError):
        model.predict(X_test)
