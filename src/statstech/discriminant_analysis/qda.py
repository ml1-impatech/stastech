"""Análise discriminante quadrática (QDA) para classificação."""

import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    Classificador de Análise Discriminante Quadrática (QDA).

    Modela cada classe por meio de uma distribuição gaussiana própria,
    estimando um vetor de médias e uma matriz de covariância para
    cada classe.

    Parameters
    ----------
    priors : array-like de formato (n_classes,), default=None
        Probabilidades a priori das classes. Se None, são calculadas
        a partir das frequências das classes nos dados de treinamento.
    reg_param : float, default=1e-6
        Parâmetro de regularização não negativo adicionado à diagonal
        das matrizes de covariância.

    Attributes
    ----------
    classes_ : ndarray de formato (n_classes,)
        Rótulos das classes encontradas durante o ajuste.

    priors_ : ndarray de formato (n_classes,)
        Probabilidades a priori estimadas para cada classe.

    means_ : ndarray de formato (n_classes, n_features)
        Vetor de médias estimado para cada classe.

    covariances_ : ndarray de formato (n_classes, n_features, n_features)
        Matrizes de covariância estimadas para as classes, com
        regularização aplicada.
    """

    def __init__(self, priors=None, reg_param=1e-6):
        """
        Inicializa o classificador QDA.

        Parameters
        ----------
        priors : array-like de formato (n_classes,), default=None
            Probabilidades a priori das classes. Se None, são calculadas
            a partir das frequências das classes nos dados de treinamento.
        reg_param : float, default=1e-6
            Valor não negativo adicionado à diagonal das matrizes
            de covariância durante o ajuste.
        """
        self.priors = priors
        self.reg_param = reg_param

    def fit(self, X, y):
        """
        Ajusta o classificador QDA aos dados de treinamento.

        Estima as probabilidades a priori das classes, seus vetores
        de médias e suas matrizes de covariância individuais.

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Amostras de treinamento.

        y : array-like de formato (n_samples,)
            Rótulos das classes correspondentes às amostras.

        Returns
        -------
        self : QuadraticDiscriminantAnalysis
            Instância do classificador ajustada aos dados.

        Raises
        ------
        ValueError
            Se os dados forem inválidos, houver menos de duas classes,
            alguma classe tiver menos de duas observações ou reg_param
            não for finito e não negativo, ou priors tiver formato
            inválido, valores não positivos ou não finitos, ou soma
            diferente de 1.
        """
        X, y = check_X_y(X, y)
        if (
            not isinstance(
                self.reg_param,
                (int, float, np.integer, np.floating),
            )
            or isinstance(self.reg_param, (bool, np.bool_))
            or not np.isfinite(self.reg_param)
            or self.reg_param < 0
        ):
            raise ValueError("reg_param deve ser um valor finito e não negativo.")

        _, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("O QDA precisa de pelo menos duas classes.")

        if self.priors is None:
            self.priors_ = np.array([np.mean(y == label) for label in self.classes_])
        else:
            try:
                priors = np.asarray(self.priors, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(
                    "priors deve conter valores numéricos positivos e finitos."
                ) from None

            if (
                priors.shape != (n_classes,)
                or not np.all(np.isfinite(priors))
                or np.any(priors <= 0)
                or not np.isclose(np.sum(priors), 1.0)
            ):
                raise ValueError(
                    "priors deve conter um valor positivo "
                    "e finito por classe, "
                    "com soma igual a 1."
                )

            self.priors_ = priors.copy()

        self.means_ = np.empty((n_classes, n_features))
        self.covariances_ = np.empty((n_classes, n_features, n_features))

        self.covariance_ranks_ = np.empty(n_classes, dtype=int)
        self.log_determinants_ = np.empty(n_classes)
        self.determinant_signs_ = np.empty(n_classes)

        for k, label in enumerate(self.classes_):
            X_k = X[y == label]
            n_k = X_k.shape[0]

            if n_k < 2:
                raise ValueError("Cada classe deve ter pelo menos duas observações.")

            self.means_[k] = np.mean(X_k, axis=0)

            X_centered = X_k - self.means_[k]
            covariance = (X_centered.T @ X_centered) / (n_k - 1)

            covariance += self.reg_param * np.eye(n_features)

            self.covariances_[k] = covariance
            self.covariance_ranks_[k] = np.linalg.matrix_rank(covariance)

            sign, log_det = np.linalg.slogdet(covariance)
            self.determinant_signs_[k] = sign
            self.log_determinants_[k] = log_det

        return self

    def predict_proba(self, X):
        """
        Estima as probabilidades de cada classe para as amostras fornecidas.

        As probabilidades são calculadas a partir das funções discriminantes
        gaussianas de cada classe e normalizadas para que cada linha some 1.

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Amostras para as quais as probabilidades serão estimadas.

        Returns
        -------
        probabilities : ndarray de formato (n_samples, n_classes)
            Probabilidades estimadas de cada amostra pertencer a cada classe.
            A soma das probabilidades de cada linha é igual a 1.

        Raises
        ------
        NotFittedError
            Se o classificador ainda não tiver sido ajustado.
        ValueError
            Se X for inválido, tiver número de características diferente
            do usado no treinamento ou se alguma matriz de covariância
            for numericamente singular ou tiver determinante não positivo.
        """
        check_is_fitted(
            self,
            attributes=[
                "classes_",
                "priors_",
                "means_",
                "covariances_",
                "covariance_ranks_",
                "log_determinants_",
                "determinant_signs_",
            ],
        )
        X = check_array(X)

        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "X deve ter o mesmo número de características usado no fit."
            )

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        scores = np.empty((n_samples, n_classes))

        for k in range(n_classes):
            centered = X - self.means_[k]
            covariance = self.covariances_[k]

            if self.covariance_ranks_[k] < covariance.shape[0]:
                raise ValueError("A matriz de covariância deve ser não singular.")

            if self.determinant_signs_[k] <= 0:
                raise ValueError(
                    "A matriz de covariância deve ter determinante positivo."
                )

            log_det = self.log_determinants_[k]
            solved = np.linalg.solve(covariance, centered.T)
            quadratic = np.sum(centered * solved.T, axis=1)

            scores[:, k] = -0.5 * quadratic - 0.5 * log_det + np.log(self.priors_[k])

        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        """
        Prediz os rótulos das classes das amostras fornecidas.

        Cada amostra é atribuída à classe com a maior probabilidade
        estimada pelo método predict_proba.

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Amostras que serão classificadas.

        Returns
        -------
        y_pred : ndarray de formato (n_samples,)
            Rótulo predito para cada amostra.

        Raises
        ------
        NotFittedError
            Se o classificador ainda não tiver sido ajustado.
        ValueError
            Se X for inválido, tiver número de características diferente
            do usado no treinamento ou se alguma matriz de covariância
            for numericamente singular ou tiver determinante não positivo.
        """
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]
