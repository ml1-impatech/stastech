import numpy as np
from  base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (check_X_y, check_array, check_is_fitted)

class Ridge(BaseEstimator, RegressorMixin):
    """
    Modelo de Regressão Linear com a regularização L2.

    O modelo Ridge estima os coeficientes que minimizam a soma dos quadrados
    dos resíduos acrescida de um termo de penalização L2. Essa penalização 
    reduz a magnitude dos coeficientes e pode ajudar a controlar o sobreajuste.

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

    n_features_in_ : int
        Número de variáveis utilizadas para calcular o modelo.
    """

    def __init__(self,
                alpha : float = 1.0,
                fit_intercept : bool = True) -> None:
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha deve ser um valor finito não negativo")

        if not isinstance(self.fit_intercept, (bool, np.bool)):
            raise ValueError("fit_intercept deve ser booleano")
        

    def fit(self, X : np.ndarray, y : np.ndarray) -> "Ridge":
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
        X, y = check_X_y(X, y)


        if self.fit_intercept is True:
             X_mean = np.mean(X, axis = 0)
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

        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept is True:
             self.intercept_ = y_mean - X_mean @ self.coef_
        else:
             self.intercept_ = 0.0

        return self      

    def predict(self, X : np.ndarray) -> np.ndarray:
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
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X deve possuir o mesmo número de variáveis que em fit")

        y_predicted = X @ self.coef_ + self.intercept_

        return y_predicted