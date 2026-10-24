"""Exemplo: análise discriminante linear para classificação."""

import numpy as np

from statstech.discriminant_analysis.lda import LinearDiscriminantAnalysis

# Dados
X = np.array(
    [
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ]
)

y = np.array([0, 0, 0, 1, 1, 1])

# Treina o modelo
model = LinearDiscriminantAnalysis()
model.fit(X, y)

# Faz previsões e calcula as probabilidades.
y_pred = model.predict(X)
probabilidades = model.predict_proba(X)

print("Classes:", model.classes_)
print("Classes reais:", y)
print("Classes previstas:", y_pred)
print("Probabilidades previstas:")
print(probabilidades)

print("Acurácia:", np.mean(y_pred == y))
