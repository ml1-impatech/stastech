"""Testes de StandardScaler e MinMaxScaler."""

import numpy as np
import pytest

from statstech.preprocessing.scalers import MinMaxScaler, StandardScaler
from statstech.utils.validation import NotFittedError


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 4.0, 30.0], size=(50, 3))


# ---------------- StandardScaler ----------------


def test_standard_mean_0_and_std_1(X):
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0)


def test_standard_hand_made_example():
    # média 2, desvio populacional 1 -> z = -1, 0, 1 (para 1, 2, 3)
    # desvio = sqrt(((1-2)^2 + 0 + (3-2)^2)/3) = sqrt(2/3)
    scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(scaler.mean_, [2.0])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(2 / 3)])
    np.testing.assert_allclose(scaler.transform([[2.0]]), [[0.0]], atol=1e-12)


def test_standard_inverse_transform(X):
    scaler = StandardScaler().fit(X)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)


def test_standard_uses_training_statistics_on_test():
    train = np.array([[0.0], [10.0]])  # média 5, desvio 5
    test = np.array([[20.0]])
    scaler = StandardScaler().fit(train)
    # (20 - 5) / 5 = 3, e não 0 (que seria se recalculasse no teste)
    np.testing.assert_allclose(scaler.transform(test), [[3.0]])


def test_standard_constant_column_does_not_give_nan():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    Z = StandardScaler().fit_transform(X)
    assert not np.isnan(Z).any()
    np.testing.assert_allclose(Z[:, 1], 0.0)


def test_standard_with_mean_and_with_std_false():
    X = np.array([[1.0], [3.0]])  # média 2, desvio 1
    np.testing.assert_allclose(
        StandardScaler(with_mean=False).fit_transform(X), [[1.0], [3.0]]
    )
    np.testing.assert_allclose(
        StandardScaler(with_std=False).fit_transform(X), [[-1.0], [1.0]]
    )


def test_standard_does_not_change_original_x(X):
    X_copy = X.copy()
    StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(X, X_copy)


def test_standard_attributes(X):
    scaler = StandardScaler().fit(X)
    assert scaler.mean_.shape == (3,)
    assert scaler.scale_.shape == (3,)
    assert scaler.n_features_in_ == 3


def test_standard_transform_before_fit():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.zeros((2, 2)))


def test_standard_different_number_of_columns(X):
    scaler = StandardScaler().fit(X)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.zeros((2, 5)))


def test_standard_invalid_input():
    with pytest.raises(ValueError, match="2D"):
        StandardScaler().fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        StandardScaler().fit([[1.0], [np.nan]])


# ---------------- MinMaxScaler ----------------


def test_minmax_goes_from_0_to_1(X):
    Z = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0)
    np.testing.assert_allclose(Z.max(axis=0), 1.0)


def test_minmax_hand_made_example():
    scaler = MinMaxScaler().fit([[0.0], [5.0], [10.0]])
    np.testing.assert_allclose(scaler.data_min_, [0.0])
    np.testing.assert_allclose(scaler.data_max_, [10.0])
    np.testing.assert_allclose(scaler.transform([[2.5]]), [[0.25]])


def test_minmax_custom_feature_range():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Z = scaler.fit_transform([[0.0], [5.0], [10.0]])
    np.testing.assert_allclose(Z.ravel(), [-1.0, 0.0, 1.0])


def test_minmax_inverse_transform(X):
    scaler = MinMaxScaler(feature_range=(2, 5)).fit(X)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)


def test_minmax_test_outside_training_range_is_not_clipped():
    scaler = MinMaxScaler().fit([[0.0], [10.0]])
    np.testing.assert_allclose(scaler.transform([[20.0], [-10.0]]), [[2.0], [-1.0]])


def test_minmax_constant_column_does_not_give_nan():
    Z = MinMaxScaler().fit_transform([[3.0], [3.0], [3.0]])
    assert not np.isnan(Z).any()


def test_minmax_invalid_feature_range():
    with pytest.raises(ValueError, match="mínimo deve ser menor"):
        MinMaxScaler(feature_range=(1, 1)).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="tupla"):
        MinMaxScaler(feature_range=5).fit([[0.0], [1.0]])


def test_minmax_before_fit_and_wrong_number_of_columns(X):
    with pytest.raises(NotFittedError):
        MinMaxScaler().transform(X)
    scaler = MinMaxScaler().fit(X)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.zeros((2, 1)))
