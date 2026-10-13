import numpy as np
from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import check_array, check_is_fitted

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        max_components = min(X.shape)
        if self.n_components is None:
            k = max_components
        elif self.n_components <= max_components and self.n_components >= 1:
            k = self.n_components
        else:
            raise ValueError('n_components fora do intervalo aceito')
        self.components_ = Vt[:k]
        self.n_components_ = k
        return self

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, Z):
        raise NotImplementedError