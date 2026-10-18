import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin
from statstech.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        classes, counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape
        n_classes = len(classes)

        means = []

        for classe in classes:
            X_classe = X[y == classe]
            media = X_classe.mean(axis=0)
            means.append(media)

        means = np.array(means)

        if self.priors is None:
            priors = counts / n_samples
        else:
            priors = np.asarray(self.priors, dtype=float)

        covariance = np.zeros((n_features, n_features))
        
        for i in range(len(classes)):
            classe = classes[i]
            mean = means[i]

            X_classe = X[y == classe]
            centered = X_classe - mean
            covariance += centered.T @ centered

        covariance /= n_samples - n_classes

        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.covariance_ = covariance

        return self