import numpy as np
from statstech.base import BaseEstimator
from statstech.utils.distances import pairwise_distances
from statstech.utils.validation import check_array, check_is_fitted

class KMeans(BaseEstimator):
    """Agrupamento k-means pelo algoritmo de Lloyd.
    Divide os dados em n_clusters grupos, alternando entre atribuir cada
    ponto ao centro mais próximo e mover cada centro para a média dos
    seus pontos. O algoritmo roda n_init vezes, com inícios diferentes,
    e fica com a rodada de menor inércia.
    O k-means usa distância euclidiana e é sensível à escala das
    features. Se elas estiverem em escalas diferentes, use o
    StandardScaler antes.

    Decisões de implementação:
    - Cluster vazio: o centro é movido para o ponto mais distante do
      seu centro atual.
    - Critério de parada: a soma dos deslocamentos ao quadrado dos
      centros entre duas iterações fica menor ou igual a tol.

    Parameters
    ----------
    n_clusters : int, default=8
        Número de grupos (k). Deve estar entre 1 e o número de amostras.
    n_init : int, default=10
        Quantas vezes rodar o algoritmo com inícios diferentes.
    max_iter : int, default=300
        Número máximo de iterações em cada rodada.
    tol : float, default=1e-4
        Tolerância do critério de parada. Deve ser maior ou igual a 0.
    init : {"k-means++", "random"}, default="k-means++"
        Como escolher os centros iniciais:
        "random" - k pontos de X sorteados sem reposição.
        "k-means++" - o primeiro centro é sorteado; cada próximo é
        sorteado com probabilidade proporcional à distância ao
        quadrado até o centro mais próximo já escolhido.
    random_state : int ou None, default=None
        Semente do gerador aleatório, para resultados reprodutíveis.

    Attributes
    ----------
    cluster_centers_ : ndarray de formato (n_clusters, n_features)
        Centros dos grupos.
    labels_ : ndarray de formato (n_amostras,)
        Grupo de cada ponto de treino.
    inertia_ : float
        Soma das distâncias ao quadrado de cada ponto até o seu centro.
        Quanto menor, mais compactos os grupos.
    n_iter_ : int
        Número de iterações da rodada escolhida.
    n_features_in_ : int
        Número de features vistas no fit.
    """
    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        init="k-means++",
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def _init_random(self, X, rng):
        """Sorteia n_clusters pontos de X como centros iniciais."""
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices]

    def _init_kmeanspp(self, X, rng):
        """Escolhe os centros iniciais pelo método k-means++."""
        indice_init = rng.choice(X.shape[0])
        indices = []
        indices.append(indice_init)
        for i in range(1, self.n_clusters):
            distances = pairwise_distances(X, X[indices], squared=True)
            d2 = np.min(distances, axis=1)
            sum_d2 = np.sum(d2)
            if sum_d2 == 0:
                next_indice = rng.choice(X.shape[0])
            else:
                probs = d2/sum_d2
                next_indice = rng.choice(X.shape[0], p=probs)
            indices.append(next_indice)
        return X[indices]

    def _lloyd(self, X, centers):
        """Roda o algoritmo de Lloyd a partir de centers.
        Devolve labels, centros finais, número de iterações e inércia.
        """
        n_iter = 0
        for _ in range (self.max_iter):
            distances = pairwise_distances(X, centers, squared=True)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            min_distances = np.min(distances, axis=1)
            for j in range(self.n_clusters):
                if (labels == j).any():
                    new_centers[j] = np.mean(X[labels==j], axis=0)
                else:
                    far = np.argmax(min_distances)
                    new_centers[j] = X[far]
                    min_distances[far] = 0
            mov = np.sum((new_centers - centers) ** 2)
            n_iter += 1
            centers = new_centers
            if mov <= self.tol:
                break
        new_distances = pairwise_distances(X, centers, squared=True)
        inertia = np.sum(np.min(new_distances, axis=1))
        labels = np.argmin(new_distances, axis=1)
        return labels, centers, n_iter, inertia

    def fit(self, X, y=None):
        """Agrupa os dados de X.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : None
            Ignorado. Existe só para manter o padrão.

        Returns
        -------
        self : KMeans
            O próprio modelo, treinado.

        Raises
        ------
        ValueError
            Se X for inválido ou se algum hiperparâmetro estiver fora
            do intervalo aceito.
        """
        X = check_array(X)
        X = X.astype(float)
        if ((not isinstance(self.n_clusters, (int, np.integer)) 
            or isinstance(self.n_clusters, bool)) or self.n_clusters < 1 
            or self.n_clusters>len(X)):
            raise ValueError("O número de clusters precisa ser "
                             f"inteiro entre 1 e {len(X)}")
        if (not isinstance(self.n_init, (int, np.integer)) 
            or isinstance(self.n_init, bool)) or self.n_init<1:
            raise ValueError("n_init precisa ser inteiro e pelo menos 1")
        if (not isinstance(self.max_iter, (int, np.integer)) or
            isinstance(self.max_iter, bool)) or self.max_iter<1:
            raise ValueError("max_iter precisa ser inteiro e pelo menos 1")
        if self.init != "random" and self.init != "k-means++":
            raise ValueError("init deve ser 'random' ou 'k-means++'")
        if (not isinstance(self.tol, (int, float, np.integer, np.floating))
            or isinstance(self.tol, bool) or self.tol < 0):
            raise ValueError("tol deve ser um número maior ou igual a zero")
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        for i in range(self.n_init):
            if self.init == "random":
                init_centers = self._init_random(X, rng)
            else:
                init_centers = self._init_kmeanspp(X, rng)
            labels, centers, n_iter, inertia = self._lloyd(X, init_centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best = [labels, centers, n_iter, inertia]
        self.labels_ = best[0]
        self.cluster_centers_ = best[1]
        self.n_iter_ = best[2]
        self.inertia_ = best[3]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Atribui cada ponto de X ao centro mais próximo.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino ou novos.

        Returns
        -------
        labels : ndarray de formato (n_amostras,)
            Índice do grupo de cada ponto.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        ValueError
            Se X tiver um número de features diferente do fit.
        """
        attributes = ["labels_", "cluster_centers_", "n_iter_",
                       "inertia_", "n_features_in_"]
        check_is_fitted(self, attributes)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features "
                f"mas deveria ter {self.n_features_in_}")
        new_distances = pairwise_distances(X, self.cluster_centers_,
                                            squared=True)
        labels = np.argmin(new_distances, axis=1)
        return labels

    def fit_predict(self, X, y=None):
        """Agrupa X e devolve o grupo de cada ponto.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : None
            Ignorado. Existe só para manter o padrão.

        Returns
        -------
        labels : ndarray de formato (n_amostras,)
            Índice do grupo de cada ponto.
        """
        return self.fit(X).labels_
