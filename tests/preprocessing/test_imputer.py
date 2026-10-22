"""Testes do SimpleImputer."""

import numpy as np
import pytest

from statstech.preprocessing.imputer import SimpleImputer
from statstech.utils.validation import NotFittedError

NAN = np.nan


@pytest.fixture
def X():
    return np.array(
        [
            [1.0, 10.0],
            [2.0, NAN],
            [NAN, 30.0],
            [5.0, 10.0],
        ]
    )


def test_mean_fills_with_mean(X):
    imputer = SimpleImputer(strategy="mean").fit(X)
    np.testing.assert_allclose(imputer.statistics_, [8 / 3, 50 / 3])
    Z = imputer.transform(X)
    np.testing.assert_allclose(Z[2, 0], 8 / 3)
    np.testing.assert_allclose(Z[1, 1], 50 / 3)


def test_median_fills_with_median(X):
    imputer = SimpleImputer(strategy="median").fit(X)
    np.testing.assert_allclose(imputer.statistics_, [2.0, 10.0])


def test_most_frequent_fills_with_most_common_value(X):
    imputer = SimpleImputer(strategy="most_frequent").fit(X)
    # coluna 1: 10 aparece duas vezes. Coluna 0: tudo único, menor = 1.
    np.testing.assert_allclose(imputer.statistics_, [1.0, 10.0])


def test_no_nan_left(X):
    for strat in ("mean", "median", "most_frequent"):
        Z = SimpleImputer(strategy=strat).fit_transform(X)
        assert not np.isnan(Z).any()


def test_values_that_were_not_nan_stay_equal(X):
    Z = SimpleImputer().fit_transform(X)
    present = ~np.isnan(X)
    np.testing.assert_array_equal(Z[present], X[present])


def test_does_not_change_original_x(X):
    X_copy = X.copy()
    SimpleImputer().fit_transform(X)
    np.testing.assert_array_equal(X, X_copy)  # NaN == NaN neste assert


def test_uses_training_values_on_test():
    train = np.array([[2.0], [4.0]])  # média 3
    test = np.array([[NAN], [100.0]])
    imputer = SimpleImputer().fit(train)
    np.testing.assert_allclose(imputer.transform(test), [[3.0], [100.0]])


def test_all_nan_column_raises_error():
    X = np.array([[1.0, NAN], [2.0, NAN]])
    with pytest.raises(ValueError, match="só têm NaN"):
        SimpleImputer().fit(X)


def test_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer(strategy="moda").fit([[1.0], [2.0]])


def test_infinity_raises_error():
    with pytest.raises(ValueError, match="infinitos"):
        SimpleImputer().fit([[1.0], [np.inf]])


def test_before_fit_and_wrong_number_of_columns(X):
    with pytest.raises(NotFittedError):
        SimpleImputer().transform(X)
    imputer = SimpleImputer().fit(X)
    with pytest.raises(ValueError, match="features"):
        imputer.transform(np.zeros((2, 3)))
