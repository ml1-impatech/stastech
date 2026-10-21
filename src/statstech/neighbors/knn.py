"""K vizinhos mais próximos (KNN): 1ª versão, com laços simples."""

import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statstech.utils.distances import pairwise_distances
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


class _BaseKNN(BaseEstimator):
    """Parte comum: guardar o treino e achar os vizinhos."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _fit_base(self, X, y):
        X, y = check_X_y(X, y)
        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError("n_neighbors deve ser um número inteiro.")
        if k < 1:
            raise ValueError("n_neighbors deve ser pelo menos 1.")
        if k > X.shape[0]:
            raise ValueError(
                f"n_neighbors={k} é maior que o número de amostras "
                f"de treino ({X.shape[0]})."
            )
        self.X_train_ = X.astype(float)
        self.y_train_ = y.copy()

    def _kneighbors(self, X):
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                "X deve ter o mesmo número de features usado no "
                f"treino ({self.X_train_.shape[1]}), mas tem {X.shape[1]}."
            )
        distances = pairwise_distances(X, self.X_train_)

        all_neighbors = []
        # Para cada ponto novo, ordeno os pontos de treino por distância
        # e fico com os k primeiros.
        for i in range(X.shape[0]):
            order = np.argsort(distances[i], kind="stable")
            all_neighbors.append(order[: self.n_neighbors])
        return np.array(all_neighbors)


class KNeighborsClassifier(_BaseKNN, ClassifierMixin):
    """Classificador dos k vizinhos mais próximos."""

    def fit(self, X, y):
        self._fit_base(X, y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X):
        neighbors = self._kneighbors(X)
        probabilities = []
        for row in neighbors:
            labels = self.y_train_[row]
            line = []
            for c in self.classes_:
                votes = 0
                for label in labels:
                    if label == c:
                        votes += 1
                line.append(votes / self.n_neighbors)
            probabilities.append(line)
        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = []
        for line in probabilities:
            predictions.append(self.classes_[np.argmax(line)])
        return np.array(predictions)


class KNeighborsRegressor(_BaseKNN, RegressorMixin):
    """Regressor dos k vizinhos mais próximos."""

    def fit(self, X, y):
        self._fit_base(X, y)
        if not np.issubdtype(self.y_train_.dtype, np.number):
            raise ValueError("y deve ser numérico na regressão.")
        self.y_train_ = self.y_train_.astype(float)
        return self

    def predict(self, X):
        neighbors = self._kneighbors(X)
        predictions = []
        for row in neighbors:
            predictions.append(np.mean(self.y_train_[row]))
        return np.array(predictions)
