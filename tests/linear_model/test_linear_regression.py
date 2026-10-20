import numpy as np
import pytest

from statstech.linear_model.linear_regresion import LinearRegression

def test_fit_noiseless_data_coefficients_and_intercept():
    X = np.array([[1, 2],
                 [2, 1],
                 [3, 4],
                 [4, 3],
                 [5, 2]],
                 dtype = float)
    
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5

    model = LinearRegression()
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, [3, -2], atol = 1e-8)
    np.testing.assert_allclose(model.interept_, 5, atol = 1e-8)


def test_result_matches_numpy_lstsq():
    X = np.array([[1, 2],
                 [2, 1],
                 [3, 4],
                 [4, 3],
                 [5, 2]],
                 dtype = float)

    y = np.array([4, 7, 8, 11, 16], dtype = float)

    model = LinearRegression()
    model.fit(X, y)

    X_augmented = np.column_stack([np.ones(X.shape[0]), X])
    coef_ref = np.linalg.lstsq(X_augmented, y, rcond = None)[0]

    np.testing.assert_allclose(model.intercept_, coef_ref[0], atol = 1e-8)
    np.testing.assert_allclose(model.coef_, coef_ref[1, :], atol = 1e-8)


def test_fit_intercept_false_through_origin():
    X = np.array([[1],
                [2],
                [3],
                [4]], 
                dtype = float)

    y = np.array([3, 6, 9, 12], dtype = float)

    model = LinearRegression(fit_intercept = False)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, [3], atol = 1e-8)
    assert model.intercept_ == 0
    

def test_predict_before_fit_raises_error():
    X = np.array([[1],
                  [2],
                  [3]],)
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.predict(X)
    

def test_mismatched_shapes_raises_value_error():
    X = np.array([[4, 6],
                  [5, 9],
                  [2, 5]],
                  dtype = float)

    y = np.array([4, 5, 9, 3], dtype = float)

    model = LinearRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)