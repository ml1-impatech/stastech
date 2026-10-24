"""Exemplo: busca de hiperparâmetros com GridSearchCV.

O GridSearchCV testa todas as combinações de uma grade de valores,
avalia cada uma por validação cruzada e guarda a melhor. Aqui ele é
usado em duas tarefas: regressão (erro quadrático médio) e
classificação (log_loss). Nas duas, quanto menor a métrica, melhor.
"""

import numpy as np

from statstech.linear_model.logistic_regression import LogisticRegression
from statstech.metrics.regression import mean_squared_error
from statstech.model_selection.search import GridSearchCV
from statstech.model_selection.split import KFold, train_test_split
from statstech.neural_network import MLPRegressor

rng = np.random.default_rng(0)

# Regressão: y = 2*x0 + 3*x1 + ruído, com a rede neural.
X = rng.normal(size=(150, 2))
y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * rng.normal(size=150)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_grid = {"learning_rate": [1e-3, 1e-2, 1e-1], "hidden_layer_sizes": [(4,), (16,)]}
grid = GridSearchCV(
    MLPRegressor(max_iter=200, random_state=0), param_grid, cv=3, scoring="mse"
)
grid.fit(X_train, y_train)

print("Regressão (scoring='mse')")
for result in grid.cv_results_:
    print(f"  {result['params']} -> mse médio {result['mean_score']:.4f}")
print("Melhores parâmetros:", grid.best_params_)
print("Melhor mse (validação cruzada):", round(grid.best_score_, 4))

# Com refit=True, best_estimator_ já foi treinado com todos os dados
# de treino, e predict usa esse modelo.
y_pred = grid.predict(X_test)
print("MSE no teste:", round(mean_squared_error(y_test, y_pred), 4))
print("R2 no teste (score do modelo):", round(grid.score(X_test, y_test), 4))

# Classificação: duas classes desbalanceadas, com a regressão logística.
X = np.vstack([rng.normal(0.0, size=(90, 2)), rng.normal(2.0, size=(30, 2))])
y = np.array([0] * 90 + [1] * 30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

grid = GridSearchCV(
    LogisticRegression(),
    {"max_iter": [1, 50, 1000]},
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    scoring="log_loss",
)
grid.fit(X_train, y_train)

print("\nClassificação (scoring='log_loss')")
for result in grid.cv_results_:
    print(f"  {result['params']} -> log_loss médio {result['mean_score']:.4f}")
print("Melhores parâmetros:", grid.best_params_)
print("Acurácia no teste:", round(grid.score(X_test, y_test), 3))

# O modelo original não é alterado: a busca trabalha com cópias.
print("\nModelo original treinado?", hasattr(grid.estimator, "classes_"))
