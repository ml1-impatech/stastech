import numpy as np
from base import BaseEstimator, RegressorMixin

class LinearRegression( BaseEstimator, RegressorMixin):

    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if y.ndim != 1:
            raise ValueError("y deve ser um vetor")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem possuir o mesmo número de amostras")

        if self.fit_intercept is True:
            X_mean = np.mean(X, axis = 0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

        else: 
            X_centered = X
            y_centered = y

        A = X_centered.T @ X_centered
        b = X_centered.T @ y_centered

        self.coef_ = np.linalg.solve(A, b)
        if self.fit_intercept is True:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if not hasattr(self, "coef_"):
            raise ValueError("O modelo não foi treinado") 

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted