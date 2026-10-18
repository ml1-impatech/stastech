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

        for class_label in classes:
            X_class = X[y == class_label]
            mean = X_class.mean(axis=0)
            means.append(mean)

        means = np.array(means)

        if self.priors is None:
            priors = counts / n_samples
        else:
            priors = np.asarray(self.priors, dtype=float)

        covariance = np.zeros((n_features, n_features))
        
        for i in range(len(classes)):
            class_label = classes[i]
            mean = means[i]

            X_class = X[y == class_label]
            centered = X_class - mean
            covariance += centered.T @ centered

        covariance /= n_samples - n_classes

        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.covariance_ = covariance

        return self

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(
            self,
            ["classes_", "priors_", "means_", "covariance_"]
        )

        scores = []

        for i in range(len(self.classes_)):
            mean = self.means_[i]
            prior = self.priors_[i]

            score = (
                X @ np.linalg.solve(self.covariance_, mean)
                - 0.5 * mean @ np.linalg.solve(self.covariance_, mean)
                + np.log(prior)
            )

            scores.append(score)

        scores = np.array(scores).T

        scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities