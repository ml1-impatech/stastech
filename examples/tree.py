"""Exemplo: árvores de decisão e overfitting.
 
Mostra como a profundidade controla o ajuste, para regressão e classificação.
Rodar com python examples/tree.py
O gráfico só aparece se o matplotlib estiver instalado.
"""
 
import numpy as np
 
from statstech.src.statstech.tree.decision_tree import (DecisionTreeClassifier,
                                                        DecisionTreeRegressor)

def split(X, y, test_size=0.3, seed=0):
    """Divisão treino/teste simples (trocar por train_test_split depois)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

# Regressão

# Classificação

# Rótulos

# Gráfico