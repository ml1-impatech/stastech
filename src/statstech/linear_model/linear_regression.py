import numpy as np
from base import BaseEstimator, RegressorMixin

class LinearRegression( BaseEstimator, RegressorMixin):
    """ 
    Modelo de regressão linear baseado em mínimos quadrados.

    Parameters
    ----------
    fit_intercept : bool, default = True
        Quando verdadeiro, o intercepto é calculado. Quando é falso,
        o intercepto não é utilizado nos cálculos.

    Attributes
    ---------
    coef_ : ndarray
        Coeficientes estimados para as variváeis de entrada.

    intercept_ : float
        Intercepto calculado pelo modelo.
    
    n_features_in_ : int
        Número de variáveis usadas para calcular o modelo.
    """

    def __init__(self, fit_intercept : bool = True) -> None:
        
        self.fit_intercept = fit_intercept

    def fit(self, X : np.ndarray, y : np.ndarray) -> "LinearRegression":
        """ 
        Ajusta o modelo aos dados de treinamento

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

        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if y.ndim != 1:
            raise ValueError("y deve ser um vetor")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem possuir o mesmo número de amostras")

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
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if not hasattr(self, "coef_"):
            raise ValueError("O modelo não foi treinado") 

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted