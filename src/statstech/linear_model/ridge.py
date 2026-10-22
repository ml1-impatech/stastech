"""Regressão linear com penalização L2"""

import numpy as np

from statstech.base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)


class Ridge(BaseEstimator, RegressorMixin):
    """
    Modelo de Regressão Linear com a regularização L2.

    O modelo Ridge estima os coeficientes que minimizam a soma dos quadrados
    dos resíduos acrescida de um termo de penalização L2. Essa penalização
    reduz a magnitude dos coeficientes e pode ajudar a controlar o sobreajuste.
    O modelo não padroniza os dados, sendo necessário usar StandardScaler antes.

    Parameters
    ----------
    alpha: float, defalt=1.0
        Constante não negativa de intensidade da penalização dos coeficientes
        que multiplica o termo de regularização L2.


    fit_intercept: bool, defalt=True
        Se True, o intercepto é calculado. Se False,
        o intercepto não é utilizado nos cálculos.


    Attributes
    ---------

    coef_: ndarray
        Coeficientes estimados para as variáveis de entrada.

    intercept_: float
        Intercepto calculado pelo modelo.

    n_features_in_: int
        Número de variáveis utilizadas para calcular o modelo.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:

        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _check_params(self):
        """
        Valida os hiperparâmetros do modelo.

        Raises
        ------
        ValueError
            Se alpha não for um valor finito não negativo.
            Se intercept não for booleano.
        """

        if (
            not isinstance(self.alpha, (int, float, np.integer, np.floating))
            or isinstance(self.alpha, (bool, np.bool_))
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError("alpha deve ser um valor finito não negativo")

        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept deve ser booleano")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        """
        Ajusta o modelo de regressão Ridge aos dados de treino.

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Matriz com os dados de treinamento.

        y : array-like de formato (n_sample,)
            Valores observados usados para ajuste do modelo.

        Returns
        -------
        self : object
            Retorna a instância ajustada pelo modelo.

        Raises
        ------
        ValueError
            Se X não for uma matriz 2D.
            Se y não for um vetor.
            Se o número de amostras de X e y não são iguais.
        """
        self._check_params()

        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

        else:
            X_centered = X
            y_centered = y

        n_features = X.shape[1]
        self.n_features_in_ = n_features

        I = np.eye(n_features)

        A = X_centered.T @ X_centered + self.alpha * I
        b = X_centered.T @ y_centered

        self.coef_, *_ = np.linalg.lstsq(A, b, rcond=None)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza previsões de y com base nos valores fornecidos em X.

        Parameters
        ----------
        X: array-like de formato (n_samples, n_features)
            Amostras a serem previstas.

        Returns
        -------
        y_predicted: ndarray de formato (n_samples,)
            Valores preditos pelo modelo.

        Raises
        ------
        ValueError
            Se X não é uma matriz bidimensional.
            Se o modelo não tiver sido treinado.
            Se o número de features de X for diferente do usado em fit.

        """
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])

        X = check_array(X)

        check_n_features(X, self.n_features_in_)

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted
