import numpy as np
import pytest

from statstech.linear_model.lasso import Lasso
from statstech.linear_model.linear_regression import LinearRegression

def test_lasso_max_alpha_zeros_all_coefficients():
    X = np.array([[1, 2], 
                  [2, 1],
                  [3, 4],
                  [4, 3],
                  [5, 2]],
                  dtype = float)

    y = np.array([4, 7, 8, 11, 16], dtype = float)

    X_centered = X - np.mean(X, axis = 0)
    y_centered = y - np.mean(y)
    n_samples = X.shape[0]

    alpha_max = np.max(np.abs(X_centered.T @ y_centered))/n_samples
    
    model = Lasso(alpha = alpha_max)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, np.zeros(X.shape[1]), atol = 1e-8)

def test_lasso_very_small_alpha_approximates_ordinary_least_squares():
    X = np.array([[1], [2], [3], [4]], dtype = float)

    y = np.array([3, 5, 7, 9], dtype = float)

    model_lasso = Lasso(alpha = 1e-8)
    model_linear_regression = LinearRegression()

    model_lasso.fit(X, y)
    model_linear_regression.fit(X, y)

    np.testing.assert_allclose(model_lasso.coef_, model_linear_regression.coef_, atol = 1e-6)
    np.testing.assert_allclose(model_lasso.intercept_, model_linear_regression.intercept_, atol = 1e-6)

def test_lasso_features_selection_zeros_irrelevant_features():
    X = np.array([[-2, 1],
                  [-1, -2],
                  [1, 2],
                  [2, -1]], 
                  dtype = float)

    y = np.array([-6, -3, 3, 6], dtype = float)

    model = Lasso()
    model.fit(X, y)

    assert abs(model.coef_[0]) > 0
    np.testing.assert_allclose(model.coef_[1], 0, atol = 1e-8)

def test_lasso_soft_threshold_function_with_manual_values():
    threshold = 2.0

    assert Lasso._sort_threshold(5.0, threshold) == 3.0
    assert Lasso._sort_threshold(-5.0, threshold) == -3.0
    assert Lasso._sort_threshold(1.0, threshold) == 0.0
    assert Lasso._sort_threshold(-1.0, threshold) == 0.0
    assert Lasso._sort_threshold(2.0, threshold) == 0.0
    assert Lasso._sort_threshold(-2.0, threshold) == 0.0

def test_lasso_defaut_parameter():
    model = Lasso()

    assert model.alpha == 1.0
    assert model.fit_intercept is True
    assert model.tol == 1e-4
    assert model.max_iter == 1000

def test_lasso_invalid_alpha_raises_value_error():
    with pytest.raises(ValueError):
        Lasso(alpha = -0.1)

def test_lasso_invalid_tol_raises_value_error():
    with pytest.raises(ValueError):
        Lasso(tol = -0.1)

def test_lasso_invalid_max_iter_raises_value_error():
    with pytest.raises(ValueError):
        Lasso(max_iter = -1)

def test_lasso_invalid_fit_intercept_raises_value_error():
    with pytest.raises(ValueError):
        Lasso(fit_intercept = 0)

def test_lasso_fit_returs_self():
    X = np.array([[1, 2], 
                  [2, 1],
                  [3, 4],
                  [4, 3],],
                  dtype = float)

    y = np.array([4, 7, 8, 11], dtype = float)

    model = Lasso()

    assert model.fit(X, y) is model

def test_lasso_fit_without_intercept():
    X = np.array([[1], 
                  [2],
                  [3]],
                  dtype = float)

    y = np.array([2, 4, 6], dtype = float)

    model = Lasso(fit_intercept = False, tol = 1e-10)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, [27 / 14])
    assert model.intercept_ == 0.0

def test_lasso_fit_multiple_features():
    X = np.array([[1, 2], 
                  [2, 1],
                  [3, 4],
                  [4, 3]],
                  dtype = float)

    y = np.array([4, 7, 8, 11], dtype = float)

    model = Lasso(alpha = 0.1)
    model.fit(X, y)

    assert model.coef_.shape == (2,)
    assert model.n_features_in_ == 2
    assert np.isfinite(model.coef_).all()

def test_lasso_fit_inconsistent_samples():
    X = np.array([[1], 
                  [2],
                  [3],
                  [4]],
                  dtype = float)

    y = np.array([4, 7], dtype = float)

    model = Lasso()

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_lasso_predict():
    X = np.array([[1],
                  [2],
                  [3]], 
                  dtype = float)
    y = np.array([3, 5, 7], dtype = float)

    model = Lasso(alpha = 0.0, tol = 1e-10)
    model.fit(X, y)

    X_test = np.array([[4],
                       [5]], 
                       dtype = float)
    pred = model.predict(X_test)

    np.testing.assert_allclose(pred, [9, 11])

def test_lasso_predict_without_fit():
    model = Lasso()
    X = np.array([[1], [2]], dtype = float)

    with pytest.raises(ValueError):
        model.predict(X)

def test_lasso_predict_wrong_number_of_features():
    X_train = np.array([[1, 2],
                        [2, 3],
                        [3, 4]], 
                        dtype = float)
    
    y_train = np.array([5, 8, 11], dtype = float)

    model = Lasso()
    model.fit(X_train, y_train)

    X_test = np.array([[4]], dtype = float)

    with pytest.raises(ValueError):
        model.predict(X_test)
