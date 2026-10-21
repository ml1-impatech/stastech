"""Exemplo: KNN para classificação e regressão (com padronização)."""

import numpy as np

from statstech.neighbors.knn import KNeighborsClassifier, KNeighborsRegressor
from statstech.preprocessing.scalers import StandardScaler
from statstech.utils.distances import pairwise_distances

# ---- Classificação ----
# Dois grupos de pontos: perto de (0, 0) e perto de (5, 5).
X = np.array([
    [0.0, 0.0], [0.5, 0.2], [0.1, 0.8],
    [5.0, 5.0], [5.5, 4.8], [4.7, 5.3],
])
y = np.array([0, 0, 0, 1, 1, 1])

# O KNN usa distâncias, então padronizamos antes.
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = KNeighborsClassifier(n_neighbors=3).fit(X_scaled, y)

X_new = scaler.transform(np.array([[0.3, 0.3], [5.2, 5.1]]))
print("Classes previstas:", model.predict(X_new))
print("Probabilidades:\n", model.predict_proba(X_new))
print("Acurácia no treino:", model.score(X_scaled, y))

# ---- Regressão ----
# y = 2x: a previsão é a média do y dos vizinhos.
X_reg = np.arange(10, dtype=float).reshape(-1, 1)
y_reg = 2 * X_reg.ravel()

regressor = KNeighborsRegressor(n_neighbors=2).fit(X_reg, y_reg)
print("Previsão para x=4.4:", regressor.predict(np.array([[4.4]])))

# ---- Distâncias entre pontos ----
print("Distâncias:\n", pairwise_distances([[0, 0], [1, 1]], [[3, 4]]))
