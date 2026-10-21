"""Exemplo: validação cruzada com KFold, LeaveOneOut e cross_val_score.
 
Mostra como os divisores geram as partições e como avaliar um modelo
com cross_val_score: com o score padrão do modelo, com o nome de uma
métrica, com uma função própria e com o log_loss (que usa as
probabilidades previstas).
"""

import numpy as np

from statstech.linear_model.logistic_regression import LogisticRegression
from statstech.model_selection.split import KFold, LeaveOneOut
from statstech.model_selection.validation import cross_val_score
from statstech.neural_network import MLPRegressor

# Divisores: índices de teste de cada partição para 10 amostras.
X_small = np.arange(10).reshape(-1, 1)

print("KFold(n_splits=5, shuffle=True)")
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_idx, test_idx) in enumerate(kfold.split(X_small), start=1):
    print(f"  fold {i}: teste = {sorted(test_idx.tolist())}")

n_partitions = sum(1 for _ in LeaveOneOut().split(X_small))
print("LeaveOneOut gera", n_partitions, "partições (uma por amostra)")

# Dados de classificação: 90 amostras da classe 0 e 30 da classe 1.
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0.0, size=(90,2)), rng.normal(2.0, size=(30, 2))])
y = np.array([0] * 90 + [1] * 30)
order = rng.permutation(len(y))
X, y = X[order], y[order]

model = LogisticRegression(max_iter=1000)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Sem scoring, usa o score do próprio modelo (acurácia).
scores = cross_val_score(model, X, y, cv=cv)
print("\nAcurácia por fold:", np.round(scores, 3))
print("Média:", round(scores.mean(), 3))

# Com scoring por nome: a métrica é escolhida pelo nome.
for name in ["accuracy", "recall_macro", "f1_macro", "log_loss"]:
    scores = cross_val_score(model, X, y, cv=cv, scoring=name)
    print(f"scoring={name!r}: média = {scores.mean():.3f}")


# Com uma função própria: recebe (y_true, y_pred) e devolve um número.
def taxa_de_erro(y_true, y_pred):
    return np.mean(y_true != y_pred)


scores = cross_val_score(model, X, y, cv=cv, scoring=taxa_de_erro)
print("\nTaxa de erro (função própria):", round(scores.mean(), 3))

# LeaveOneOut: um score por amostra (aqui, com as 30 primeiras).
scores = cross_val_score(model, X[:30], y[:30], cv=LeaveOneOut())
print("\nLeaveOneOut:", len(scores), "scores, média", round(scores.mean(), 3))

# O modelo original não é treinado: a validação cruzada usa cópias.
print("Modelo original treinado?", hasattr(model, "classes_"))

# Regressão com a rede neural: o erro quadrático médio é "menor é melhor".
X_reg = rng.normal(size=(150, 2))
y_reg = 2* X_reg[:, 0] + 3 * X_reg[:, 1] + 0.1 * rng.normal(size=150)

mlp = MLPRegressor(
    hidden_layer_sizes=(16,), learning_rate=1e-2, max_iter=200, random_state=0
)
scores = cross_val_score(mlp, X_reg, y_reg, cv=3, scoring="mse")
print("\nMLPRegressor, mse por fold:", np.round(scores, 4))