import numpy as np
import pytest

from statstech.preprocessing.polynomial import PolynomialFeatures
from statstech.linear_model.linear_regression import LinearRegression

def test_linear_regression_on_poly_features_matches_polyfit():
    X = np.array([[1],
                  [2],
                  [3],
                  [4]],
                  dtype = float)

    y = np.array([2, 5, 10, 17], dtype = float)

    model_poly = PolynomialFeatures()
    X_poly = model_poly.fit_transform(X)

    linear_model = LinearRegression()
    linear_model.fit(X_poly, y)

    coef = np.polyfit(X[:, 0], y, deg = 2)

    np.testing.assert_allclose(linear_model.coef_, coef[1::-1], atol = 1e-8)
    np.testing.assert_allclose(linear_model.intercept_, coef[-1], atol = 1e-8)
   

def test_polynomial_features_output_shape_matches_combinatorial_formula():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]],
                  dtype = float)

    deg = 3
    n_features = X.shape[1]

    poly = PolynomialFeatures(degree = deg, include_bias = False)
    X_poly = poly.fit_transform(X)

    expected_n_columns = sum(np.prod(np.arange(n_features, n_features + d))// 
                             np.prod(np.arange(1, d + 1)) 
                             for d in range(1, deg + 1))

    assert X_poly.shape == (X.shape[0], expected_n_columns)

def test_polynomial_features_small_manual_example_degree_two():
    X = np.array([[2, 3]], dtype = float)

    poly = PolynomialFeatures()
    X_poly = poly.fit_transform(X)

    expected = np.array([[2, 3, 4, 6, 9]], dtype = float)

    np.testing.assert_array_equal(X_poly, expected)

def test_polynomial_features_include_bias():
    X = np.array([[2, 3],
                  [4, 5]],
                  dtype = float)
    
    poly = PolynomialFeatures(include_bias = True)
    X_poly = poly.fit_transform(X)

    expected = np.array([[1, 2, 3, 4, 6, 9],
                        [1, 4, 5, 16, 20, 25]],
                        dtype = float)

    np.testing.assert_array_equal(X_poly, expected)

def test_polynomial_features_without_bias():
    X = np.array([[2, 3]], dtype = float)

    poly = PolynomialFeatures()
    X_poly = poly.fit_transform(X)

    assert X_poly.shape == (1, 5)
    np.testing.assert_array_equal(X_poly, np.array([[2, 3, 4, 6, 9]], dtype = float))

def test_polynomial_features_invalid_degree_raises_value_error():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree = -1)

def test_polynomial_features_invalide_include_bias_raises_value_error():
    with pytest.raises(ValueError):
        PolynomialFeatures(include_bias = 0)

def test_polynomial_features_transform_without_fit_raises_error():
    poly = PolynomialFeatures()

    X = np.array([[1, 2],
                  [3, 4]],
                  dtype = float)

    with pytest.raises(ValueError):
        poly.transform(X)

def test_polynomial_features_wrong_number_of_features_raises_value_error():
    X_train = np.array([[1, 2],
                        [3, 4]],
                        dtype = float)
    X_test = np.array([[1], [2]], dtype = float)

    poly = PolynomialFeatures()
    poly.fit(X_train)

    with pytest.raises(ValueError):
        poly.transform(X_test)

def test_polynomial_features_fit_returns_self():
    X = np.array([[1, 2],
                  [3, 4]], 
                  dtype = float)
    
    poly = PolynomialFeatures()

    assert poly.fit(X) is poly

def test_polynomial_features_fit_transform_matches_fit_and_transform():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], 
                  dtype = float)

    poly_1 = PolynomialFeatures(degree = 3)
    X_poly_1 = poly_1.fit_tranform(X)

    poly_2 = PolynomialFeatures(degree = 3)
    poly_2 = poly_2.fit(X)
    X_poly_2 = poly_2.transform(X)

    np.testing.assert_array_equal(X_poly_1, X_poly_2)
    

def test_polynomial_features_degree_one():
    X = np.array([[1, 2], [3, 4]], dtype = float)

    poly = PolynomialFeatures(degree = 1)
    X_poly = poly.fit_transform(X)

    np.testing.assert_array_equal(X_poly, X)