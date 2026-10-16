import numpy as np
import itertools
from base import BaseEstimator

class PolinomialFeatures(BaseEstimator):
    def __init__(self, degree = 2, include_bias = False):
        self.degree = degree
        self.include_bias = include_bias
        
    def fit(self, X, y = None):
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode estar vazio")

        self.n_features_in_ = X.shape[1]

        self.combinations_ = []
        for d in range(1, self.degree + 1):
            for i in itertools.combinations_with_replacement(range(self.n_features_in_), d):
                self.combinations_.append(i)

        n_out = len(self.combinations_)

        if self.include_bias is True:
            n_out += 1

        self.n_output_features = n_out

        return self


    def transform(self, X):
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if not hasattr(self, "combinations_"):
            raise ValueError("O modelo não foi treinado")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("O número de variáveis deve ser igual ao usado em fit")

        n_samples, n_features = X.shape
        columns = []

        if self.include_bias is True:
            columns.append(np.ones(n_samples))

        for i in self.combinations_:
            column = np.prod(X[:, i], axis = 1)
            columns.append(column)

        return np.column_stack(columns)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)