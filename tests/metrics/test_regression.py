import numpy as np
import pytest

from statstech.metrics.regression import (
    _validate_regression_inputs,
    residual_sum_squares,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)




def test_validate_inputs_returns_2d_float64_arrays(y_true, y_pred):
    result_y_true, result_y_pred = _validate_regression_inputs(y_true, y_pred)

    assert isinstance(result_y_pred, np.ndarray)
    assert isinstance(result_y_true, np.ndarray)

    assert result_y_true.dtype == np.float64
    assert result_y_pred.dtype == np.float64

    assert result_y_true.ndim == 1
    assert result_y_pred.ndim == 1

    np.testing.assert_array_equal(result_y_true, np.asarray(y_true, dtype=np.float64))
    np.testing.assert_array_equal(result_y_pred, np.asarray(y_pred, dtype=np.float64))

def test_validate_inputs_rejects_different_shapes():
    y_true = [13, 22]
    y_pred = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        _validate_regression_inputs(y_true, y_pred)

def test_validate_inputs_rejects_multidimensional_shapes():
    y_true = [[13, 22], [15, 36]]
    y_pred = [[25, 40], [14, 12]]

    with pytest.raises(ValueError):
        _validate_regression_inputs(y_true, y_pred)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([], [4, 5, 6]),
        ([1, 2, 3], []),
    ],
)

def test_validate_inputs_rejects_empty_shapes(y_true, y_pred):
    with pytest.raises(ValueError):
        _validate_regression_inputs(y_true, y_pred)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([20, 2, np.inf], [4, 5, 6]),
        ([1, 2, 3], [3, 10, np.inf]),
    ],
)

def test_validate_inputs_rejects_infinite(y_true, y_pred):
    with pytest.raises(ValueError):
        _validate_regression_inputs(y_true, y_pred)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([20, 2, np.nan], [4, 5, 6]),
        ([1, 2, 3], [3, 10, np.nan]),
    ],
)

def test_validate_inputs_rejects_nan(y_true, y_pred):
    with pytest.raises(ValueError):
        _validate_regression_inputs(y_true, y_pred)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_residual_sum_squares_accepts_valid_arrays(y_true, y_pred):
    
    assert isinstance(residual_sum_squares(y_true, y_pred), float)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_residual_sum_squares_returns_expected_value(y_true, y_pred):
    
    assert residual_sum_squares(y_true, y_pred) == 27.0

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_mean_squared_error_returns_expected_value(y_true, y_pred):

    assert mean_squared_error(y_true, y_pred) == 9.0

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_mean_absolute_error_returns_expected_value(y_true, y_pred):

    assert mean_absolute_error(y_true, y_pred) == 3.0

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_root_mean_squared_error_returns_expected_value(y_true, y_pred):

    assert root_mean_squared_error(y_true, y_pred) == 3.0

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        (np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32),
        ),
    ],
)

def test_r2_score_returns_expected_value(y_true, y_pred):

    assert r2_score(y_true, y_pred) == -12.5

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([2, 2, 2], [4, 5, 6]),
        ((2, 2, 2), (4, 4, 4)),
        (np.array([3, 3, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),),
        (
            np.array([4, 4, 4], dtype=np.int32),
            np.array([4, 4, 4], dtype=np.float32),
        ),
    ],
)

def test_r2_score_raises_value_error_when_y_true_is_constant(y_true, y_pred):

    with pytest.raises(ValueError):
        r2_score(y_true, y_pred)