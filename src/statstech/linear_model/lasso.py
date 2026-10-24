"""Regressão linear com penalização L1"""

import numpy as np

from statstech.base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)


class Lasso(BaseEstimator, RegressorMixin):
    """
    Modelo de Regressão Linear com regularização L1.

    O modelo Lasso estima os coeficientes que minimizam a soma dos quadrados
    dos resíduos acrescida de um termo de penalização L1. Essa penalização
    pode levar alguns coeficientes a valerem zero, realizando seleção de
    variáveis

    Parameters
    ----------
    alpha: float, default=1.0
        Constante não negativa que multiplica o termo de penalização L1,
        determinando a intensidade da regularização.

    fit_intercept: bool, default=True
        Quando verdadeiro, o intercepto é calculado. Quando é falso, o
        intercepto não é usado nos cálculos.

    tol: float, default=1e-4
        Tolerância usada para verificar a otimização dos coeficientes.

    max_iter: int, default=1000
        Número máximo de iterações para o algoritmo.

    Attributes
    ----------
    coef_: ndarray
        Coeficientes estimados para as variáveis de entrada.

    intercept_: float
        Intercepto calculado pelo modelo.

    n_iter_: int
        Número de iterações.

    n_features_in_: int
        Número de variáveis calculada pelo modelo.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        max_iter: int = 1000,
    ) -> None:

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter

    def _check_params(self):
        """
        Valida os hiperparâmetros do modelo.

        Raises
        ------
        ValueError
            Se alpha não for um valor finito não negativo.
            Se fit_intercept não for booleano.
            Se tol não for um valor finito não negativo.
            Se max_iter não for um valor inteiro positivo.
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

        if (
            not isinstance(self.tol, (int, float, np.integer, np.floating))
            or not np.isfinite(self.tol)
            or self.tol < 0
        ):
            raise ValueError("tol deve ser um valor finito não negativo")

        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter deve ser um valor inteiro positivo")

    @staticmethod
    def _soft_threshold(z: float, threshold: float) -> float:
        """
        Aplica o operador de limiarização suave.

        Parameters
        ----------
        z: float
            Valor em que o operador está sendo aplicado.

        threshold: float
            Limiar utilizado na transformação.

        Returns
        -------
        float
            Valor resultante do limiar suave.
        """

        if z > threshold:
            return z - threshold
        elif z < -threshold:
            return z + threshold
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        """
        Ajusta o modelo de regressão Lasso aos dados de treino.

        Parameters
        ----------
        X: array-like de formato (n_samples, n_features)
            Matriz com os dados de treinamento.

        y: array-like de formato (n_samples,)
            Valores observados usados para o ajuste do modelo.

        Returns
        -------
        self: object
            Retorna a instância ajustada pelo modelo.

        Raises
        ------
        ValueError
            Se X não for uma matriz bidimensional.
            Se y não for um vetor unidimensional.
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
            X_centered = X.copy()
            y_centered = y.copy()

        n_samples, n_features = X_centered.shape

        self.n_features_in_ = n_features

        coef = np.zeros(n_features)

        col_norms = np.sum(X_centered**2, axis=0)

        threshold = self.alpha * n_samples

        for iteration in range(1, self.max_iter + 1):
            max_change = 0.0

            for i in range(n_features):
                if col_norms[i] == 0:
                    coef[i] = 0.0
                    continue

                old_coef = coef[i]

                residual = y_centered - X_centered @ coef + X_centered[:, i] * old_coef

                rho = X_centered[:, i] @ residual

                new_coef = self._soft_threshold(rho, threshold) / col_norms[i]

                coef[i] = new_coef

                change = abs(new_coef - old_coef)
                max_change = max(max_change, change)

            if max_change < self.tol:
                break

        self.coef_ = coef
        self.n_iter_ = iteration

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
            Se X não é uma matriz 2D.
            Se o modelo não tiver sido treinado.
            Se o número de features de X for diferente do usado em fit.
        """

        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])

        X = check_array(X)

        check_n_features(X, self.n_features_in_)

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted
