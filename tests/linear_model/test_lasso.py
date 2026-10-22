import numpy as np
import pytest

from statstech.src.linear_model.lasso import Lasso
from statstech.src.linear_model.linear_regression import LinearRegression

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
    X = np.array([[]], dtype = float)

    y = np.array([], dtype = float)

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
    pass

def test_lasso_invalid_alpha_raises_value_error():
    pass

def test_lasso_invalid_tol_raises_value_error():
    pass

def test_lasso_invalid_max_iter_raises_value_error():
    pass

def test_lasso_invalid_fit_intercept_raises_value_error():
    pass

def test_lasso_fit_returs_self():
    pass

def test_lasso_fit_without_intercept():
    pass

def test_lasso_fit_inconsistent_samples():
    pass

def test_lasso_predict():
    pass

def test_lasso_predict_without_fit():
    pass

def test_lasso_predict_wrong_number_of_features():
    pass