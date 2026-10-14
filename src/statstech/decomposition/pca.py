import numpy as np
from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import check_array, check_is_fitted

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        """ treinamento do modelo"""
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
        max_abs_Vt = np.argmax(np.abs(Vt), axis = 1)
        lines = np.arange(Vt.shape[0])
        columns = max_abs_Vt
        values = Vt[lines, columns]
        signs = np.sign(values)
        Vt = Vt * signs[:, np.newaxis]
        self.components_ = Vt[:k]
        self.n_components_ = k
        n = X.shape[0]
        variances = S**2 / (n-1)
        variances_total = np.sum(variances)
        ratios = variances/variances_total
        self.explained_variance_ = variances[:k]
        self.explained_variance_ratio_ = ratios[:k]
        return self

    def transform(self, X):
        """ Pega dados de treino ou novos e escreve cada ponto nas coordenadas
        das componentes que o fit aprendeu
        """
        attributes = ["mean_", "components_"]
        check_is_fitted(self, attributes)
        X = check_array(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError(f"X tem {X.shape[1]} features "
                            f"mas deveria ter {len(self.mean_)}")
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z):
        raise NotImplementedError