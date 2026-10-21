"""Exemplo: pré-processamento (imputação, escala e codificação)."""

import numpy as np

from statstech.preprocessing.encoders import LabelEncoder, OneHotEncoder
from statstech.preprocessing.imputer import SimpleImputer
from statstech.preprocessing.scalers import MinMaxScaler, StandardScaler

# Dados numéricos com um valor faltando (NaN).
X_treino = np.array([[1.0, 200.0], [2.0, np.nan], [3.0, 400.0]])
X_teste = np.array([[np.nan, 300.0]])

# 1) Preencher NaN: o valor vem do treino e é reaproveitado no teste.
imputer = SimpleImputer(strategy="mean").fit(X_treino)
X_treino = imputer.transform(X_treino)
X_teste = imputer.transform(X_teste)
print("Treino sem NaN:\n", X_treino)
print("Teste sem NaN:\n", X_teste)

# 2) Padronizar: média 0 e desvio 1 (ajustado só no treino).
scaler = StandardScaler().fit(X_treino)
print("Padronizado:\n", scaler.transform(X_treino))
print("Voltando ao original:\n",
      scaler.inverse_transform(scaler.transform(X_treino)))

# 3) Escalar para o intervalo [0, 1].
print("Min-max:\n", MinMaxScaler().fit_transform(X_treino))

# 4) Variáveis categóricas: colunas 0/1 (one-hot).
cores = np.array([["azul"], ["verde"], ["azul"], ["vermelho"]])
encoder = OneHotEncoder(drop="first").fit(cores)
print("Categorias:", encoder.categories_)
print("One-hot (sem a 1ª categoria):\n", encoder.transform(cores))

# 5) Rótulos do alvo y: texto -> 0, 1, ...
label = LabelEncoder()
codigos = label.fit_transform(["sim", "nao", "sim"])
print("Códigos:", codigos)
print("De volta:", label.inverse_transform(codigos))
