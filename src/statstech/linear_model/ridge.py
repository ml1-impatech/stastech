import numpy as np
from  base import BaseEstimator, RegressorMixin

class Ridge(BaseEstimator, RegressorMixin):
    """
    Modelo de Regressão Linear com a regularização L2.

    Parameters
    ----------
    alpha : float, defaut = 1.0
        Constante não negativa de intensidade da penalização dos coeficientes
        que multiplica o termo de regularização L2.


    fit_intercept : bool, defalt = True
        Quando verdadeiro, o intercepto é calculado. Quando é falso, 
        o intercepto não é utilizado nos calculos.


    Attributes
    ---------

    coef_ : ndarray
        Coeficientes estimados para as variáveis de entrada.

    intercept_ : float
        Intercepto calculado pelo modelo.

    n_features_in : int
        Número de variáveis utilizadas para calcular o modelo.
    """

    def __init__(self, alpha = 1.0, fit_intercept = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Ajusta o modelo de regressão Ridge aos dados de treino.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Matriz com os dados de treinamento.

        y : array-like, shape(n_sample,)
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
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")
        
        if y.ndim != 1:
            raise ValueError("y deve ser um vetor")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem possuir o mesmo número de amostras")

        if self.fit_intercept is True:
             X_mean = np.mean(X, axis = 0)
             y_mean = np.mean(y)

             X_centered = X - X_mean
             y_centered = y - y_mean

        else:
             X_centered = X
             y_centered = y

        n_features = X.shape[1]

        I = np.eye(n_features)

        A = X_centered.T @ y_centered + self.alpha * I
        b = X_centered.T * y_centered

        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept is True:
             self.intercept_ = y_mean - X_mean @ self.coef_
        else:
             self.intercept_ = 0.0

        return self      

    def predict(self, X):
        """
        Realiza previsões de y com base nos valores fornecidos em X.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Amostras a serem previstas.

        Returns
        -------
        y_predicted : ndarray, shape(n_samples,)
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