"""Exemplo: árvores de decisão e overfitting.
 
Mostra como a profundidade controla o ajuste, para regressão e classificação.
Rodar com python examples/tree.py
O gráfico só aparece se o matplotlib estiver instalado.
"""
 
import numpy as np
 
from statstech.metrics.regression import mean_squared_error
from statstech.model_selection.search import GridSearchCV
from statstech.model_selection.split import train_test_split
from statstech.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ----------------------------------------------------------------------
# 1. Regressão: y = sen(x) + ruído
# ----------------------------------------------------------------------
 
rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(300, 1))
y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=300)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
 
print("=== Regressão (y = sen(x) + ruído) ===")
print(f"{'max_depth':>10} {'MSE treino':>12} {'MSE teste':>12} {'prof. real':>11}")
 
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
train_err, test_err = [], []
for d in depths:
    reg = DecisionTreeRegressor(max_depth=d).fit(X_tr, y_tr)
    e_tr = mean_squared_error(y_tr, reg.predict(X_tr))
    e_te = mean_squared_error(y_te, reg.predict(X_te))
    train_err.append(e_tr)
    test_err.append(e_te)
    print(f"{d!s:>10} {e_tr:>12.4f} {e_te:>12.4f} {reg.get_depth():>11}")
 
print("\nSem max_depth o erro de treino vai a zero (a árvore decora o ruído),")
print("mas o erro de teste volta a subir: isso é overfitting.\n")

# ----------------------------------------------------------------------
# 2. Classificação: duas classes com fronteira não linear
# ----------------------------------------------------------------------
 
rng = np.random.default_rng(7)
Xc = rng.normal(size=(400, 2))
yc = ((Xc[:, 0] ** 2 + Xc[:, 1] ** 2) > 1.2).astype(int)  # anel
flip = rng.random(400) < 0.08  # 8% de ruído
yc = np.where(flip, 1 - yc, yc)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=1, stratify=yc
)
 
print("=== Classificação (fronteira circular, 8% de rótulos trocados) ===")
print(f"{'max_depth':>10} {'acc treino':>12} {'acc teste':>12} {'prof. real':>11}")
 
for d in [1, 2, 3, 4, 6, 8, None]:
    clf = DecisionTreeClassifier(max_depth=d).fit(Xc_tr, yc_tr)
    print(
        f"{d!s:>10} {clf.score(Xc_tr, yc_tr):>12.3f} "
        f"{clf.score(Xc_te, yc_te):>12.3f} {clf.get_depth():>11}"
    )


# ----------------------------------------------------------------------
# 3. Escolhendo max_depth com validação cruzada
# ----------------------------------------------------------------------
 
print("\n=== GridSearchCV para max_depth (regressão) ===")
search = GridSearchCV(
    DecisionTreeRegressor(),
    {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, None]},
    cv=5,
    scoring="mse",
).fit(X_tr, y_tr)
print("melhor max_depth:", search.best_params_["max_depth"])
print(f"MSE médio na validação cruzada: {search.best_score_:.4f}")
print(f"MSE no teste: {mean_squared_error(y_te, search.predict(X_te)):.4f}")

# ----------------------------------------------------------------------
# 3. Rótulos em texto
# ----------------------------------------------------------------------
 
print("\n=== Rótulos em texto ===")
X_txt = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
y_txt = np.array(["baixo", "baixo", "baixo", "alto", "alto", "alto"])
clf = DecisionTreeClassifier().fit(X_txt, y_txt)
print("classes_:", clf.classes_)
print("predict([[2.5], [9.0]]):", clf.predict(np.array([[2.5], [9.0]])))
 
# ----------------------------------------------------------------------
# 4. Gráfico (opcional)
# ----------------------------------------------------------------------
 
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n(matplotlib não instalado: gráfico ignorado)")
else:
    labels = [str(d) for d in depths]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
 
    ax = axes[0]
    ax.plot(labels, train_err, "o-", label="treino")
    ax.plot(labels, test_err, "o-", label="teste")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("MSE")
    ax.set_title("Overfitting na árvore de regressão")
    ax.legend()
 
    ax = axes[1]
    grid = np.linspace(-3, 3, 400).reshape(-1, 1)
    ax.scatter(X_tr[:, 0], y_tr, s=8, color="gray", alpha=0.5, label="treino")
    for d in [2, None]:
        reg = DecisionTreeRegressor(max_depth=d).fit(X_tr, y_tr)
        ax.plot(grid[:, 0], reg.predict(grid), label=f"max_depth={d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Curvas ajustadas")
    ax.legend()
 
    fig.tight_layout()
    plt.show()
 