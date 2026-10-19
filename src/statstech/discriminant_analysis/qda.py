import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin
from statstech.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, reg_param=0.0):
        self.reg_param = reg_param

    
    def fit(self, X, y):
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.priors_ = np.empty(n_classes)
        self.means_ = np.empty((n_classes, n_features))
        self.covariances_ = np.empty(
            (n_classes, n_features, n_features)
        )

        for k, classe in enumerate(self.classes_):
            X_k = X[y == classe]
            n_k = X_k.shape[0]

            self.priors_[k] = n_k / n_samples

            self.means_[k] = np.mean(X_k, axis=0)

            X_centered = X_k - self.means_[k]
            covariance = (
                X_centered.T @ X_centered
            ) / (n_k - 1)

            covariance += self.reg_param * np.eye(n_features)

            self.covariances_[k] = covariance

        return self
