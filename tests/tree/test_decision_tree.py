"""Testes das árvores de decisão"""

import numpy as np
import pytest

from statstech.src.statstech.base import clone
from statstech.src.statstech.tree.decision_tree import (DecisionTreeClassifier , 
                                          DecisionTreeRegressor, 
                                          _Node,
                                          _node_depth)


# ----------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------

def count_splits(node):
    """Número de nós internos da subárvore."""
    if node.feature is None:
        return 0
    return 1 + count_splits(node.left) + count_splits(node.right)


@pytest.fixture
def step_data():
    """Uma feature; uma única divisão (entre 3 e 10) separa tudo."""
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=200) > 0)
    return X, y.astype(int)


# ----------------------------------------------------------------------
# _Node e _node_depth
# ----------------------------------------------------------------------
 
def test_node_depth_leaf_is_zero():
    assert _node_depth(_Node(value=1.0)) == 0

def test_node_depth_counts_longest_path():
    leaf = _Node(value=0.0)
    left = _Node(feature=0, threshold=1.0, left=leaf, right=leaf)
    root = _Node(feature=0, threshold=2.0, left=left, right=leaf)
    assert _node_depth(root) == 2


# ----------------------------------------------------------------------
# Impureza e valor das folhas (feitos à mão)
# ----------------------------------------------------------------------
 
def test_gini_pure_node_is_zero():
    clf = DecisionTreeClassifier()
    assert clf._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)

def test_gini_by_hand():
    clf = DecisionTreeClassifier()
    # p = (0.5, 0.5) -> 1 - (0.25 + 0.25) = 0.5
    assert clf._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    # p = (0.75, 0.25) -> 1 - (0.5625 + 0.0625) = 0.375
    assert clf._impurity(np.array([0, 0, 0, 1])) == pytest.approx(0.375)

def test_classifier_leaf_value_majority_and_tie():
    clf = DecisionTreeClassifier()
    assert clf._leaf_value(np.array([0, 1, 1])) == 1
    # empate -> menor índice
    assert clf._leaf_value(np.array([0, 0, 1, 1])) == 0

def test_regressor_impurity_is_variance():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array(([1, 1, 1]))
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.tree_.feature is None
    assert clf.get_depth() == 0
    np.testing.assert_array_equal(clf.predict(X), y)

def test_regressor_leaf_value_is_mean():
    reg = DecisionTreeRegressor()
    assert reg._leaf_value(np.array([1.0, 2.0, 6.0])) == pytest.approx(3.0)

 
# ----------------------------------------------------------------------
# Estrutura da árvore
# ----------------------------------------------------------------------

def test_pure_node_becomes_leaf():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.tree_.feature is None
    assert clf.get_depth() == 0
    np.testing.assert_array_equal(clf.predict(X), y)

def test_single_split_separates_perfectly(step_data):
    X, y = step_data
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.tree_.feature == 0
    assert clf.tree_.threshold == pytest.approx(6.5)
    assert clf.get_depth() == 1
    assert clf.score(X, y) == 1.0

def test_max_depth_1_makes_exactly_one_split(noisy_data):
    X, y = noisy_data
    clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert count_splits(clf.tree_) == 1
    assert clf.get_depth() == 1


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5])
def test_depth_never_exceeds_max_depth(noisy_data, max_depth):
    X, y = noisy_data
    clf = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
    assert clf.get_depth() <=max_depth



def test_min_samples_split_blocks_division(step_data):
    X, y = step_data
    clf = DecisionTreeClassifier(min_samples_split=7).fit(X, y)
    assert clf.tree_.feature is None  # 6 amostras < 7 -> folha
 
 
def test_unrestricted_tree_memorizes_training_set(noisy_data):
    X, y = noisy_data
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.score(X, y) == 1.0
 
 
def test_identical_rows_with_different_labels_do_not_crash():
    X = np.ones((4, 2))
    y = np.array([0, 1, 0, 1])
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.tree_.feature is None
    assert clf.predict(X).shape == (4,)


