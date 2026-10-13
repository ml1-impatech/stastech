import numpy as np
from  base import BaseEstimator, RegressorMixin

class Ridge(BaseEstimator, RegressorMixin):

    def __init__(self, alpha = 1.0, fit_intercept = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        
        if y.ndim != 1:
            raise ValueError("y must be a vector")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.fit_intercept is True:
             X_mean = np.mean(X, axis = 0)
             y_mean = np.mean(y)

             X_centered = X - X_mean
             y_centered = y - y_mean

        else:
             X_centered = X
             y_centered = y

        n_features = X.shape[1]

        I = np.eye(n_features)

        A = X_centered.T @ y_centered + self.alpha * I
        b = X_centered.T * y_centered

        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept is True:
             self.intercept_ = y_mean - X_mean @ self.coef_
        else:
             self.intercept_ = 0.0

        return self      

    def predict(self, X):
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        if not hasattr(self, "coef_"):
            raise ValueError("The model has not been trained")

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted