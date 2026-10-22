import numpy as np

from statstech.decomposition.pca import PCA

rng = np.random.default_rng(0)

# Dados 2D: x2 depende de x1, então os pontos ficam numa nuvem alongada
x1 = rng.normal(size=100)
x2 = 2 * x1 + rng.normal(scale=0.5, size=100)
X = np.column_stack([x1, x2])

pca = PCA().fit(X)

print("Componentes (uma por linha):\n", np.round(pca.components_, 3))
print("Variância explicada:", np.round(pca.explained_variance_, 3))
print("Fração da variância:", np.round(pca.explained_variance_ratio_, 3))
print("Direção esperada, (1, 2)/raiz de 5:", np.round(np.array([1, 2]) / 5**0.5, 3))

# Reduzir de 5 para 2 dimensões
X5 = rng.normal(size=(100, 5)) * [5.0, 3.0, 1.0, 0.5, 0.1]
pca = PCA(n_components=2).fit(X5)
Z = pca.transform(X5)

print("\nFormato antes:", X5.shape, "depois:", Z.shape)
print(
    "Fração explicada pelas 2 componentes:",
    round(pca.explained_variance_ratio_.sum(), 3),
)

# Escolher pela fração, o menor número de componentes que chega a 95%
pca = PCA(n_components=0.95).fit(X5)

print("\nComponentes para 95%:", pca.n_components_)
print("Fração acumulada:", np.round(np.cumsum(pca.explained_variance_ratio_), 3))

# Reconstrução: voltar ao espaço original perde o que foi descartado
print()
for k in [1, 2, 5]:
    pca = PCA(n_components=k).fit(X5)
    X_back = pca.inverse_transform(pca.transform(X5))
    error = np.mean((X5 - X_back) ** 2)
    print(f"Erro de reconstrução com {k} componente(s):", round(error, 4))

# Dados novos são centralizados com a média do treino
pca = PCA(n_components=2).fit(X5)
x_new = pca.mean_.reshape(1, -1)

print("\nA média do treino, transformada:", np.round(pca.transform(x_new), 3))
