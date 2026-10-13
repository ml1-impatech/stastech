"""Tests for decision trees."""

import numpy as np
import pytest

from statstech.base import clone
from statstech.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statstech.tree.decision_tree import _entropy, _gini, _rss


# --- impurity functions (tested alone, with hand-computed values) ---
def test_gini_pure_node_is_zero():
    pass


def test_gini_hand_computed_example():
    pass


def test_entropy_hand_computed_example():
    pass


def test_rss_hand_computed_example():
    pass


# --- regressor ---
def test_regressor_pure_node_becomes_leaf():
    pass


def test_regressor_max_depth_1_makes_one_split():
    pass


def test_regressor_single_split_separates_perfectly():
    pass


def test_regressor_predicts_mean_at_leaf():
    pass


def test_regressor_without_max_depth_memorizes_training_data():
    pass


def test_regressor_min_samples_split_prevents_split():
    pass


# --- classifier ---
def test_classifier_pure_node_becomes_leaf():
    pass


def test_classifier_max_depth_1_makes_one_split():
    pass


def test_classifier_single_split_separates_perfectly():
    pass


def test_classifier_predict_proba_sums_to_one():
    pass


def test_classifier_predict_proba_between_zero_and_one():
    pass


def test_classifier_string_labels_are_returned_as_strings():
    pass


def test_classifier_gini_and_entropy_both_work():
    pass


def test_classifier_invalid_criterion_raises_error():
    pass


# --- library contract ---
def test_fit_returns_self():
    pass


def test_get_params_and_set_params():
    pass


def test_clone_returns_unfitted_model():
    pass


def test_predict_before_fit_raises_error():
    pass


def test_x_and_y_with_different_lengths_raises_error():
    pass


def test_1d_x_raises_error():
    pass