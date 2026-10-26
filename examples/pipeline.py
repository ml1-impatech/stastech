"""Exemplo: Pipeline (padronização reajustada em cada fold)."""

import numpy as np

from statstech.linear_model import LogisticRegression, Ridge
from statstech.model_selection import GridSearchCV, KFold, cross_val_score
from statstech.pipeline import Pipeline
from statstech.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# ---- Regressão com validação cruzada ----
# Duas features em escalas bem diferentes.
X = rng.normal(size=(200, 2)) * [1.0, 100.0]
y = 3 * X[:, 0] + X[:, 1] / 50 + rng.normal(scale=0.5, size=200)

pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Em cada fold, o StandardScaler aprende a média e o desvio só com os
# dados de treino daquele fold: nada do fold de validação vaza.
scores = cross_val_score(pipe, X, y, cv=cv, scoring="rmse")
print("RMSE por fold:", scores.round(3))

# ---- Busca de hiperparâmetros ----
# Parâmetro de um passo: nome do passo, dois sublinhados, nome do parâmetro.
grid = GridSearchCV(pipe, {"model__alpha": [0.01, 1.0, 100.0]}, cv=cv, scoring="rmse")
grid.fit(X, y)
print("Melhor alpha:", grid.best_params_)

# Os passos ajustados ficam em named_steps_ (o pipeline original não muda).
best_model = grid.best_estimator_.named_steps_["model"]
print("Coeficientes (por desvio-padrão de cada feature):", best_model.coef_.round(3))

# ---- Classificação ----
y_class = (y > 0).astype(int)
clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
clf.fit(X, y_class)
print("Classes:", clf.classes_)
print("Probabilidades das 3 primeiras amostras:\n", clf.predict_proba(X[:3]).round(3))
print("Acurácia no treino:", clf.score(X, y_class))