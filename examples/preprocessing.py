"""Exemplo: pré-processamento (imputação, escala e codificação)."""

import numpy as np

from statstech.preprocessing.encoders import LabelEncoder, OneHotEncoder
from statstech.preprocessing.imputer import SimpleImputer
from statstech.preprocessing.scalers import MinMaxScaler, StandardScaler

# Dados numéricos com um valor faltando (NaN).
X_train = np.array([[1.0, 200.0], [2.0, np.nan], [3.0, 400.0]])
X_test = np.array([[np.nan, 300.0]])

# 1) Preencher NaN: o valor vem do treino e é reaproveitado no teste.
imputer = SimpleImputer(strategy="mean").fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
print("Treino sem NaN:\n", X_train)
print("Teste sem NaN:\n", X_test)

# 2) Padronizar: média 0 e desvio 1 (ajustado só no treino).
scaler = StandardScaler().fit(X_train)
print("Padronizado:\n", scaler.transform(X_train))
print("Voltando ao original:\n", scaler.inverse_transform(scaler.transform(X_train)))

# 3) Escalar para o intervalo [0, 1].
print("Min-max:\n", MinMaxScaler().fit_transform(X_train))

# 4) Variáveis categóricas: colunas 0/1 (one-hot).
colors = np.array([["azul"], ["verde"], ["azul"], ["vermelho"]])
encoder = OneHotEncoder(drop="first").fit(colors)
print("Categorias:", encoder.categories_)
print("One-hot (sem a 1ª categoria):\n", encoder.transform(colors))

# 5) Rótulos do alvo y: texto -> 0, 1, ...
label = LabelEncoder()
codes = label.fit_transform(["sim", "nao", "sim"])
print("Códigos:", codes)
print("De volta:", label.inverse_transform(codes))
