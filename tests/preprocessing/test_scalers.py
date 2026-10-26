"""Testes de StandardScaler e MinMaxScaler."""

import numpy as np
import pytest

from statstech.base import clone
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


# ---------------- StandardScaler com columns ----------------


def test_standard_columns_only_scales_chosen_columns(X):
    Z = StandardScaler(columns=[0, 2]).fit_transform(X)
    np.testing.assert_allclose(Z[:, [0, 2]].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z[:, [0, 2]].std(axis=0), 1.0)
    np.testing.assert_array_equal(Z[:, 1], X[:, 1])


def test_standard_columns_keeps_dummy_as_0_and_1():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(50.0, 10.0, size=40), rng.integers(0, 2, size=40)]
    Z = StandardScaler(columns=[0]).fit_transform(X)
    np.testing.assert_array_equal(Z[:, 1], X[:, 1])
    assert set(Z[:, 1]) == {0.0, 1.0}


def test_standard_columns_none_is_the_default(X):
    np.testing.assert_allclose(
        StandardScaler(columns=None).fit_transform(X),
        StandardScaler().fit_transform(X),
    )


def test_standard_columns_with_all_columns_is_the_default(X):
    np.testing.assert_allclose(
        StandardScaler(columns=[0, 1, 2]).fit_transform(X),
        StandardScaler().fit_transform(X),
    )


def test_standard_columns_order_does_not_matter(X):
    np.testing.assert_allclose(
        StandardScaler(columns=[2, 0]).fit_transform(X),
        StandardScaler(columns=[0, 2]).fit_transform(X),
    )


def test_standard_columns_accepts_numpy_array(X):
    np.testing.assert_allclose(
        StandardScaler(columns=np.array([0, 2])).fit_transform(X),
        StandardScaler(columns=[0, 2]).fit_transform(X),
    )


def test_standard_columns_attributes(X):
    scaler = StandardScaler(columns=[0]).fit(X)
    assert scaler.mean_[0] == pytest.approx(X[:, 0].mean())
    assert scaler.scale_[0] == pytest.approx(X[:, 0].std())
    np.testing.assert_array_equal(scaler.mean_[1:], 0.0)
    np.testing.assert_array_equal(scaler.scale_[1:], 1.0)


def test_standard_columns_inverse_transform(X):
    scaler = StandardScaler(columns=[1]).fit(X)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)


def test_standard_columns_uses_training_statistics_on_test():
    train = np.array([[0.0, 1.0], [10.0, 0.0]])  # coluna 0: média 5, desvio 5
    test = np.array([[20.0, 1.0]])
    scaler = StandardScaler(columns=[0]).fit(train)
    np.testing.assert_allclose(scaler.transform(test), [[3.0, 1.0]])


def test_standard_columns_constant_column_does_not_give_nan():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    Z = StandardScaler(columns=[0, 1]).fit_transform(X)
    assert not np.isnan(Z).any()
    np.testing.assert_allclose(Z[:, 1], 0.0)


def test_standard_columns_with_mean_false():
    X = np.array([[1.0, 5.0], [3.0, 6.0]])  # coluna 0: desvio 1
    Z = StandardScaler(columns=[0], with_mean=False).fit_transform(X)
    np.testing.assert_allclose(Z, [[1.0, 5.0], [3.0, 6.0]])


def test_standard_columns_not_changed_by_fit(X):
    columns = [2, 0]
    scaler = StandardScaler(columns=columns).fit(X)
    assert scaler.columns is columns
    assert columns == [2, 0]


def test_standard_columns_in_get_params_and_clone():
    scaler = StandardScaler(columns=[0, 2])
    assert scaler.get_params()["columns"] == [0, 2]
    assert clone(scaler).columns == [0, 2]


@pytest.mark.parametrize(
    "columns", [[], [3], [-1], [0, 0], [0.0], [True], "abc", [[0]]]
)
def test_standard_invalid_columns(X, columns):
    with pytest.raises(ValueError, match="columns"):
        StandardScaler(columns=columns).fit(X)


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
