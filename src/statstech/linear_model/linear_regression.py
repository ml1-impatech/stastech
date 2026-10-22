"""Regressão linear por mínimos quadrados"""

import numpy as np
from statstech.base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (check_X_y, check_array, check_is_fitted, check_n_features)

class LinearRegression(BaseEstimator, RegressorMixin):
    """ 
    Modelo de regressão linear baseado em mínimos quadrados.

    O modelo estima os coeficientes que minimizam a soma dos quadrados 
    das diferenças entre os valores observados e os valores previstos.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Se True, o intercepto é calculado. Se False,
        o intercepto não é utilizado nos cálculos.

    Attributes
    ----------
    coef_: ndarray
        Coeficientes estimados para as variáveis de entrada.

    intercept_: float
        Intercepto calculado pelo modelo.
    
    n_features_in_: int
        Número de variáveis usadas para calcular o modelo.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        
        self.fit_intercept = fit_intercept

    def _check_params(self):
        """
        Valida os hiperparâmetros do classificador.

        Raises
        ------
        ValueError
            Se fit_intercept não for booleano.
        """
    
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept deve ser booleano.")


    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """ 
        Ajusta o modelo aos dados de treinamento.

        Parameters
        ----------
        X: array-like de formato (n_samples, n_features)
            Matriz com os dados de treinamento.

        y: array-like de formato (n_samples,)
            Valores observados usados para ajuste do modelo.


        Returns
        -------
        self: object
            Retorna a instância ajustada do modelo.

        Raises
        ------
        ValueError
            Se X não for uma matriz bidimensional.
            Se y não for um vetor unidimensional.
            Se o número de amostras de X e y não são iguais.
        """
        self._check_params()
        
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

        else: 
            X_centered = X
            y_centered = y

        self.coef_, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)

        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza previsões de y com base nos valores fornecidos em X.

        Parameters
        ----------
        X: array-like de formato(n_samples, n_features)
            Amostras a serem previstas.

        Returns
        -------
        y_predict: ndarray de formato (n_samples, )
            Valores preditos pelo modelo.

        Raises
        ------
        ValueError
            Se X não é uma matriz bidimensional.
            Se X não possuir o número de variáveis do X usado em fit.
            Se o modelo não tiver sido treinado.
                
        """
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])
        X = check_array(X)

        check_n_features(X, self.n_features_in_)
        
        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted