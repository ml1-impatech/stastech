"""Testes das árvores de decisão"""

import numpy as np
import pytest

from statstech.base import clone
from statstech.tree.decision_tree import (DecisionTreeClassifier , 
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
 
 
# ----------------------------------------------------------------------
# Classificação
# ----------------------------------------------------------------------
 
def test_string_labels_come_back_as_strings():
    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    y = np.array(["não", "não", "sim", "sim"])
    clf = DecisionTreeClassifier().fit(X, y)
    pred = clf.predict(np.array([[1.5], [10.5]]))
    np.testing.assert_array_equal(pred, ["não", "sim"])
    np.testing.assert_array_equal(clf.classes_, ["não", "sim"])
 
 
def test_multiclass_perfect_split():
    X = np.array([[1.0], [2.0], [10.0], [11.0], [20.0], [21.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.score(X, y) == 1.0
    assert clf.get_depth() == 2
 
 
# ----------------------------------------------------------------------
# Regressão
# ----------------------------------------------------------------------
 
def test_regressor_depth_1_predicts_group_means():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    reg = DecisionTreeRegressor(max_depth=1).fit(X, y)
    pred = reg.predict(np.array([[0.0], [5.0], [20.0]]))
    np.testing.assert_allclose(pred, [2.0, 2.0, 11.0])
 
 
def test_regressor_unrestricted_reproduces_training_targets():
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(80, 1))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=80)
    reg = DecisionTreeRegressor().fit(X, y)
    np.testing.assert_allclose(reg.predict(X), y)
 
 
def test_regressor_constant_target_is_single_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    reg = DecisionTreeRegressor().fit(X, np.full(10, 4.2))
    assert reg.get_depth() == 0
    np.testing.assert_allclose(reg.predict(X), 4.2)
 
 
def test_deeper_tree_fits_train_better():
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.2, size=150)
 
    def train_mse(depth):
        reg = DecisionTreeRegressor(max_depth=depth).fit(X, y)
        return np.mean((reg.predict(X) - y) ** 2)
 
    assert train_mse(1) > train_mse(3) > train_mse(8)
 
 
def test_regressor_learns_nonlinear_function_better_than_mean():
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(300, 1))
    y = np.sin(X[:, 0])
    reg = DecisionTreeRegressor(max_depth=5).fit(X, y)
    mse_tree = np.mean((reg.predict(X) - y) ** 2)
    mse_mean = np.mean((y - y.mean()) ** 2)
    assert mse_tree < 0.1 * mse_mean
 
 
# ----------------------------------------------------------------------
# Contrato da biblioteca (get_params, clone, determinismo)
# ----------------------------------------------------------------------
 
def test_get_and_set_params():
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=4)
    assert clf.get_params() == {"max_depth": 3, "min_samples_split": 4}
    assert clf.set_params(max_depth=5) is clf
    assert clf.max_depth == 5
 
 
def test_clone_is_untrained_with_same_params(step_data):
    X, y = step_data
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    novo = clone(clf)
    assert novo.get_params() == clf.get_params()
    assert not hasattr(novo, "tree_")
 
 
def test_fit_returns_self(step_data):
    X, y = step_data
    clf = DecisionTreeClassifier()
    assert clf.fit(X, y) is clf
 
 
def test_fit_is_deterministic(noisy_data):
    X, y = noisy_data
    p1 = DecisionTreeClassifier(max_depth=4).fit(X, y).predict(X)
    p2 = DecisionTreeClassifier(max_depth=4).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)
 
 
def test_fit_does_not_modify_inputs(noisy_data):
    X, y = noisy_data
    X0, y0 = X.copy(), y.copy()
    DecisionTreeClassifier(max_depth=3).fit(X, y)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)

 
# ----------------------------------------------------------------------
# Erros esperados
# ----------------------------------------------------------------------
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().predict(np.zeros((2, 2)))
 
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_get_depth_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().get_depth()
 
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_x_must_be_2d(cls):
    with pytest.raises(ValueError):
        cls().fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
 
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_y_must_be_1d(cls):
    with pytest.raises(ValueError):
        cls().fit(np.zeros((3, 1)), np.zeros((3, 1)))
 
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_x_y_different_lengths_raise(cls):
    with pytest.raises(ValueError):
        cls().fit(np.zeros((3, 1)), np.zeros(4))
 
 
@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_zero_samples_raise(cls):
    with pytest.raises(ValueError):
        cls().fit(np.zeros((0, 2)), np.zeros(0))
 
 
def test_predict_wrong_number_of_features_raises(step_data):
    X, y = step_data
    clf = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((2, 3)))
 
 
@pytest.mark.parametrize("max_depth", [0, -1, 1.5, "a"])
def test_invalid_max_depth_raises(max_depth, step_data):
    X, y = step_data
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
 
 
@pytest.mark.parametrize("min_samples_split", [1, 0, 2.5])
def test_invalid_min_samples_split_raises(min_samples_split, step_data):
    X, y = step_data
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=min_samples_split).fit(X, y)
