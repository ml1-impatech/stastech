import numpy as np

from statstech.cluster.kmeans import KMeans

rng = np.random.default_rng(0)

# Dados: 3 grupos de 50 pontos em torno de centros conhecidos
true_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in true_centers])

km = KMeans(n_clusters=3, random_state=0).fit(X)

print("Centros encontrados:\n", np.round(km.cluster_centers_, 3))
print("Pontos em cada grupo:", np.bincount(km.labels_))
print("Inércia:", round(km.inertia_, 3))
print("Iterações:", km.n_iter_)

# Pontos novos vão para o centro mais próximo
X_new = np.array([[0.2, -0.1], [4.8, 5.3], [-0.3, 4.6]])

print("\nGrupo dos pontos novos:", km.predict(X_new))

print()
for k in range(1, 7):
    inertia = KMeans(n_clusters=k, random_state=0).fit(X).inertia_
    print(f"k = {k}: inércia = {inertia:.2f}")

# Inicialização com uma rodada só (n_init=1), um início ruim pode
# juntar dois grupos. O k-means++ erra menos que o sorteio simples
print()
for init in ["random", "k-means++"]:
    inertias = np.array(
        [
            KMeans(n_clusters=3, init=init, n_init=1, random_state=seed).fit(X).inertia_
            for seed in range(20)
        ]
    )
    # Uma rodada acertou se chegou perto da melhor inércia possível
    good = np.sum(inertias < 1.05 * km.inertia_)
    print(f"{init}: {good} de 20 rodadas acharam os 3 grupos")

# fit_predict: treina e devolve o grupo de cada ponto de uma vez
labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)

print("\nRótulos dos 5 primeiros pontos:", labels[:5])
