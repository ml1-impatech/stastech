import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin
from statstech.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """Classificador que utiliza análise discriminante linear.

    Parameters
    ----------
    priors : array-like, default=None
        Probabilidades a priori de cada classe.
        Se None, são calculadas a partir dos dados de treinamento.
    reg_param : float, default=1e-6
        Valor adicionado à diagonal da matriz de covariância
        para evitar problemas numéricos.
    """

    def __init__(self, priors=None, reg_param=1e-6):
        """Inicializa os parâmetros do modelo."""

        self.priors = priors
        self.reg_param = reg_param

    def fit(self, X, y):
        """Treina o modelo com os dados fornecidos.

        Parameters
        ----------
        X : array-like
            Dados de treinamento.
        y : array-like
            Classes de cada amostra.

        Returns
        -------
        self : LinearDiscriminantAnalysis
            Modelo treinado.

        Raises
        ------
        ValueError
            Se X ou y tiverem formato inválido ou dimensões incompatíveis.
            Se reg_param não for finito ou for negativo.
            Se houver menos de duas classes.
            Se o número de amostras não for maior que o número de classes.
            Se priors tiver formato inválido, valores não positivos,
            valores não finitos ou soma diferente de 1.
        """

        X, y = check_X_y(X, y)

        if (
            not isinstance(
                self.reg_param,
                (int, float, np.integer, np.floating)
            )
            or not np.isfinite(self.reg_param)
            or self.reg_param < 0
        ):
            raise ValueError(
                "reg_param deve ser um valor finito e não negativo."
            )

        classes, counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape
        n_classes = len(classes)

        if n_classes < 2:
            raise ValueError("O LDA precisa de pelo menos duas classes.")

        if n_samples <= n_classes:
            raise ValueError(
                "O número de amostras deve ser maior que "
                "o número de classes."
            )

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

            if priors.ndim != 1 or len(priors) != n_classes:
                raise ValueError(
                    "priors deve ter um valor para cada classe."
                )

            if (
                not np.isfinite(priors).all()
                or np.any(priors <= 0)
                or not np.isclose(priors.sum(), 1.0)
            ):
                raise ValueError(
                    "priors deve ter valores positivos e finitos "
                    "com soma igual a 1."
                )

        covariance = np.zeros((n_features, n_features))

        for i in range(len(classes)):
            class_label = classes[i]
            mean = means[i]

            X_class = X[y == class_label]
            centered = X_class - mean
            covariance += centered.T @ centered

        covariance /= n_samples - n_classes
        covariance += self.reg_param * np.eye(n_features)

        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.covariance_ = covariance

        return self

    def predict_proba(self, X):
        """Calcula as probabilidades previstas para cada classe.

        Parameters
        ----------
        X : array-like
            Amostras a serem classificadas.

        Returns
        -------
        probabilities : ndarray
            Probabilidades de cada amostra pertencer a cada classe.

        Raises
        ------
        ValueError
            Se X tiver formato inválido ou um número de características
            diferente do utilizado no treinamento.
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        np.linalg.LinAlgError
            Se a matriz de covariância for singular.
        """
        check_is_fitted(
            self,
            ["classes_", "priors_", "means_", "covariance_"]
        )

        X = check_array(X)

        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "X deve ter o mesmo número de características "
                "usado no treinamento."
            )

        scores = []

        for i in range(len(self.classes_)):
            mean = self.means_[i]
            prior = self.priors_[i]

            solved_mean = np.linalg.solve(self.covariance_, mean)

            score = (
                X @ solved_mean
                - 0.5 * mean @ solved_mean
                + np.log(prior)
            )

            scores.append(score)

        scores = np.array(scores).T

        scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        """Prevê a classe de cada amostra.

        Parameters
        ----------
        X : array-like
            Amostras a serem classificadas.

        Returns
        -------
        predictions : ndarray
            Classes previstas para as amostras.

        Raises
        ------
        ValueError
            Se X tiver formato inválido ou um número de características
            diferente do utilizado no treinamento.
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        np.linalg.LinAlgError
            Se a matriz de covariância for singular.
        """
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]
