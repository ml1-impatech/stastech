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

# ----------------------------------------------------------------------
# 1. Regressão: y = sen(x) + ruído
# ----------------------------------------------------------------------
 
rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(300, 1))
y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=300)
X_tr, X_te, y_tr, y_te = split(X, y)
 
print("=== Regressão (y = sen(x) + ruído) ===")
print(f"{'max_depth':>10} {'MSE treino':>12} {'MSE teste':>12} {'prof. real':>11}")
 
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
train_err, test_err = [], []
for d in depths:
    reg = DecisionTreeRegressor(max_depth=d).fit(X_tr, y_tr)
    e_tr = mse(y_tr, reg.predict(X_tr))
    e_te = mse(y_te, reg.predict(X_te))
    train_err.append(e_tr)
    test_err.append(e_te)
    print(f"{str(d):>10} {e_tr:>12.4f} {e_te:>12.4f} {reg.get_depth():>11}")
 
print("\nSem max_depth o erro de treino vai a zero (a árvore decora o ruído),")
print("mas o erro de teste volta a subir: isso é overfitting.\n")


# ----------------------------------------------------------------------
# 2. Classificação: duas classes com fronteira não linear
# ----------------------------------------------------------------------
 
rng = np.random.default_rng(7)
Xc = rng.normal(size=(400, 2))
yc = ((Xc[:, 0] ** 2 + Xc[:, 1] ** 2) > 1.2).astype(int)  # anel
flip = rng.random(400) < 0.08                             # 8% de ruído
yc = np.where(flip, 1 - yc, yc)
Xc_tr, Xc_te, yc_tr, yc_te = split(Xc, yc, seed=1)
 
print("=== Classificação (fronteira circular, 8% de rótulos trocados) ===")
print(f"{'max_depth':>10} {'acc treino':>12} {'acc teste':>12} {'prof. real':>11}")
 
for d in [1, 2, 3, 4, 6, 8, None]:
    clf = DecisionTreeClassifier(max_depth=d).fit(Xc_tr, yc_tr)
    print(
        f"{str(d):>10} {clf.score(Xc_tr, yc_tr):>12.3f} "
        f"{clf.score(Xc_te, yc_te):>12.3f} {clf.get_depth():>11}"
    )
# Rótulos

# Gráfico