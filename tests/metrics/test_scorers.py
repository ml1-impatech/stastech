import pytest

import statstech.metrics.scorers as sc
import statstech.metrics.regression as rg
import statstech.metrics.classification as cl

def test_get_scorer_returns_correct_function():
    assert sc.get_scorer("mse") == rg.mean_squared_error

def test_get_scorer_raises_value_error_for_unknown_metric():
    with pytest.raises(ValueError):
        sc.get_scorer("este_scorer_não_existe")

@pytest.mark.parametrize(
    ("metric", "exp_value"),
    [
        ("mse", True),
        ("accuracy", False),
    ],
)

def test_lower_is_better_returns_true(metric, exp_value):
    assert sc.lower_is_better(metric) == exp_value

@pytest.mark.parametrize(
    ("y_true", "y_pred", "exp_value"),
    [
        ([0,1,1], [0,1,0], 2/3),
    ],
)

def test_accuracy_scorer_calculates_correctly(y_true, y_pred, exp_value):
    assert sc.get_scorer("accuracy")(y_true,y_pred) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("metric", "exp_value"),
    [
        ('log_loss', True),
        ('mse', False),
    ],
)

def test_needs_proba(metric, exp_value):
    assert sc.needs_proba(metric) == exp_value

@pytest.mark.parametrize(
    ("y_true", "y_pred", "exp_value"),
    [
        ([0,0], [0,1], 0.5),
    ],
)

def test_accuracy_scorer_handles_predicted_class_absent_from_true(y_true, y_pred, exp_value):
    assert sc.get_scorer('accuracy')(y_true, y_pred) == exp_value