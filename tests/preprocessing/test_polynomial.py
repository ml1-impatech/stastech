import numpy as np
import pytest

from statstech.src.preprocessing.polynomial import PolinomialFeatures
from statstech.src.linear_model.linear_regression import LinearRegression

def test_linear_regression_on_poly_features_matches_polyfit():
    X = np.array([[1],
                  [2],
                  [3],
                  [4]],
                  dtype = float)

    y = np.array([2, 5, 10, 17], dtype = float)

    model_poly = PolinomialFeatures()
    X_poly = model_poly.fit_transform(X)

    linear_model = LinearRegression()
    linear_model.fit(X_poly, y)

    coef = np.polyfit(X[:, 0], y, deg = 2)

    np.testing.assert_allclose(linear_model.coef_, coef[:0:-1], atol = 1e-8)
    np.testing.assert_allclose(linear_model.intercept_, coef[-1], atol = 1e-8)
   

def test_polynomial_features_output_shape_matches_combinatorial_formula():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]],
                  dtype = float)

    deg = 3
    n_features = X.shape[1]

    poly = PolinomialFeatures(degree = deg, include_bias = False)
    X_poly = poly.fit_transform(X)

    expected_n_columns = sum(np.prod(np.arange(n_features, n_features + d))// 
                             np.prod(np.arange(1, d + 1)) 
                             for d in range(1, deg + 1))

    assert X_poly.shape == (X.shape[0], expected_n_columns)

def test_polynomial_features_small_manual_example_degree_two():
    X = np.array([[2, 3]], dtype = float)

    poly = PolinomialFeatures()
    X_poly = poly.fit_transform(X)

    expected = np.array([[2, 3, 4, 6, 9]], dtype = float)

    np.testing.assert_array_equal(X_poly, expected)
