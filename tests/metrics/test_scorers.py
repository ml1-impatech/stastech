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


def test_get_scorer_returns_classification_function():
    assert sc.get_scorer("accuracy") == cl.accuracy_score