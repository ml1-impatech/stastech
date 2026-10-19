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
 