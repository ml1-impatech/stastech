import numpy as np
from statstech.base import BaseEstimator
from statstech.utils.distances import pairwise_distances
from statstech.utils.validation import check_array, check_is_fitted

class KMeans(BaseEstimator):
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
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices]

    def _lloyd(self, X, centers):
        n_iter = 0
        for _ in range (self.max_iter):
            distances = pairwise_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for j in range(self.n_clusters):
                if (labels == j).any():
                    new_centers[j] = np.mean(X[labels==j], axis=0)
            mov = np.sum((new_centers - centers) ** 2)
            n_iter += 1
            centers = new_centers
            if mov <= self.tol:
                break
        new_distances = pairwise_distances(X, centers)
        inertia = np.sum(np.min(new_distances, axis=1))
        labels = np.argmin(new_distances, axis=1)
        return labels, centers, n_iter, inertia

    def fit(self, X, y=None):
        X = check_array(X)
        if type(self.n_clusters) is not int or self.n_clusters < 1 or self.n_clusters>len(X):
            raise ValueError(f"O número de clusters precisa ser inteiro entre 1 e {len(X)}")
        if type(self.n_init) is not int or self.n_init<1:
            raise ValueError("n_init precisa ser inteiro e pelo menos 1")
        if type(self.max_iter) is not int or self.max_iter<1:
            raise ValueError("max_iter precisa ser inteiro e pelo menos 1")
        if self.init != "random" and self.init != "k-means++":
            raise ValueError("init deve ser 'random' ou 'k-means++'")
        if self.tol<0:
            raise ValueError("tol deve ser maior que zero")

        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        for i in range(self.n_init):
            init_centers = self._init_random(X, rng)
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
        attributes = ["labels_", "cluster_centers_", "n_iter_", "inertia_", "n_features_in_"]
        check_is_fitted(self, attributes)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features mas deveria ter {self.n_features_in_}")
        new_distances = pairwise_distances(X, self.cluster_centers_)
        labels = np.argmin(new_distances, axis=1)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
