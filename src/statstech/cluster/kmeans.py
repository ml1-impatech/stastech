import numpy as np
from statstech.base import BaseEstimator
from statstech.utils.distances import pairwise_distances

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
        rng = np.random.default_rng(self.random_state)
        return None

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X, y=None):
        raise self.fit(X).labels_
