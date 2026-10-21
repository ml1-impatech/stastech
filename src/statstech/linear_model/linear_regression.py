import numpy as np
from base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (check_X_y, check_array, check_is_fitted)

class LinearRegression( BaseEstimator, RegressorMixin):
    """ 
    Modelo de regressão linear baseado em mínimos quadrados.

    O modelo estima os coeficientes que minimizam a soma dos quadrados 
    das diferenças entre os valores observados e os valores previstos.

    Parameters
    ----------
    fit_intercept : bool, default = True
        Quando verdadeiro, o intercepto é calculado. Quando é falso,
        o intercepto não é utilizado nos cálculos.

    Attributes
    ----------
    coef_ : ndarray
        Coeficientes estimados para as variváeis de entrada.

    intercept_ : float
        Intercepto calculado pelo modelo.
    
    n_features_in_ : int
        Número de variáveis usadas para calcular o modelo.
    """

    def __init__(self, fit_intercept : bool = True) -> None:
        
        self.fit_intercept = fit_intercept

        if not isinstance(self.fit_intercept, (bool, np.bool)):
            raise ValueError("fit_intercept deve ser booleano")

    def fit(self, X : np.ndarray, y : np.ndarray) -> "LinearRegression":
        """ 
        Ajusta o modelo aos dados de treinamento.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matriz com os dados de treinamento.

        y: array-like, shape (n_samples,)
            Valores observados usados para ajuste do modelo.


        Returns
        -------
        self : object
            Retorna a instância ajustada do modelo.

        Raises
        ------
        ValueError
            Se X não for uma matriz 2D.
            Se y não for um vetor.
            Se o número de amostras de X e y não são iguais.
        """

        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept is True:
            X_mean = np.mean(X, axis = 0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

        else: 
            X_centered = X
            y_centered = y

        A = X_centered.T @ X_centered
        b = X_centered.T @ y_centered

        self.coef_ = np.linalg.solve(A, b)
        if self.fit_intercept is True:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Realiza previsões de y com base nos valores fornecidos em X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Amostras a serem previstas.

        Returns
        -------
        y_predict : ndarray, shape (n_samples, )
            Valores preditos pelo modelo.

        Raises
        ------
        ValueError
            Se X não é uma matriz 2D.
            Se o modelo não tiver sido treinado.
                
        """
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X deve possuir o mesmo número de variáveis que em fit")
        
        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted