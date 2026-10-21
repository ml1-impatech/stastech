"""Exemplo: divisão em treino e teste com train_test_split.

Mostra a divisão simples, o uso do test_size como inteiro e o
parâmetro stratify, que mantém a proporção das classes no treino e
no teste mesmo quando as classes são desbalanceadas.
"""

import numpy as np

from statstech.model_selection.split import train_test_split

# Dados: 20 amostras e 2 features; 15 da classe 0 e 5 da classe 1.
rng = np.random.default_rng(0)
X = rng.normal(size=(20, 2))
y = np.array([0] * 15 + [1] * 5)

print("Classes no total (0, 1):", np.bincount(y))

# Divisão simples: 25% das amostras vão para o teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("\nDivisão simples (test_size=0.25)")
print("Formato de X_train e X_test:", X_train.shape, X_test.shape)
print("Classes no treino (0, 1):", np.bincount(y_train))
print("Classes no teste (0, 1):", np.bincount(y_test))

# test_size inteiro: número exato de amostras no teste.
_, X_test, _, y_test = train_test_split(X, y, test_size=4, random_state=0)

print("\nDivisão com test_size=4")
print("Amostras no teste:", len(X_test))

# Com stratify=y, cada classe aparece no teste na mesma proporção no total.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

print("\nDivisão estratificada (test_size=0.2)")
print("Classes no treino (0, 1):", np.bincount(y_train))
print("Classes no teste (0, 1):", np.bincount(y_test))
