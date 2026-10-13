import numpy as np
from base import BaseEstimator, RegressorMixin

class Lasso(BaseEstimator, RegressorMixin):

    def __init__(self, alpha = 1.0, fit_intercept = True, tol = 1e-4, max_iteration = 1000):
        self.alpha = alpha
        self.fit_interept = fit_intercept
        self.tol = tol
        self.max_iteration = max_iteration

    @staticmethod
    def _sort_threshold(z, threshold):
        if z > threshold:
            return z - threshold
        elif z < -threshold:
            return z + threshold
        else:
            return 0.0

    def fit(self, X, y):
        X = np.asarrya(X, dtype = float)
        y = np.asarray(y, dtype = float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        if y.ndim != 1:
            raise ValueError("y must be a vector")

        if X.shape[1] != y.shape[1]:
            raise ValueError("X and y must have the same number of sample")

        if self.fit_interept is True:
            X_mean = np.mean(X, axis = 0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

        else:
            X_centered = X.copy()
            y_centered = y.copy()

        n_sample, n_features = X_centered.shape

        coef = np.zeros(n_features)

        col_norms = np.sum(X_centered ** 2, axis = 0)

        threshold = self.alpha / 2

        for iteration in range(1 , self.max_iteration + 1):
            max_change = 0.0

            for i in range(n_features):
                if col_norms[i] == 0:
                    coef[i] = 0.0
                    continue

                old_coef = coef[i]

                residual = y_centered - X_centered @ coef + X_centered[:, i] * old_coef

                rho = X_centered[:, i] @ residual

                new_coef = self._sort_threshold(rho, threshold) / col_norms[i]

                coef[i] = new_coef

                change = abs(new_coef - old_coef)
                max_change = max(max_change, change)

                if max_change < self.tol:
                    break

            self.coef_ = coef
            self.n_iteration_ = iteration

            if self.fit_interept is True:
                self.intercept_ = (y_mean - X_mean @ self.coef )
            else:
                self.intercept_ = 0.0

            return self


    def predict(self, X):
        X = np.asarray(X, dtype = float)

        if not hasattr(self, "coef_"):
            raise ValueError("The model has not been trained")

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        y_predicted = X @ self.coef_ + self.intercept_