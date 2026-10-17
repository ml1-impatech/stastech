import numpy as np
import itertools
from base import BaseEstimator

class PolinomialFeatures(BaseEstimator):
    """
    Transforma as variáveis de entrada em características polinomiais
    para implementar a regressão polinomial.

    Parameters
    ----------
    degree : int, default = 2
        Grau máximo do polinômio gerado

    include_bias : bool, default = True
        Quando verdadeiro, adiciona a coluna do intercepto 
        prenchida com uns na matriz.
    """

    def __init__(self, degree = 2, include_bias = False):
        self.degree = degree
        self.include_bias = include_bias
        
    def fit(self, X, y = None):
        """
        Calcula as combinações de características polinomiais 
        com base dos dados de treino.

        Paramenters
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de treinamentos utilizadas para realizar o cálculo das combinações.

        y : array-like, shape (n_samples,)
            Não utilizado. Mantido por padrão de interface.

        Returns
        -------
        self : object
            Retorna a instância do próprio estimador.

        Raises
        ------
        ValueError
            Se X não for uma matriz 2D
            Se X estiver vazio
        """

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode estar vazio")

        self.n_features_in_ = X.shape[1]

        self.combinations_ = []
        for d in range(1, self.degree + 1):
            for i in itertools.combinations_with_replacement(range(self.n_features_in_), d):
                self.combinations_.append(i)

        n_out = len(self.combinations_)

        if self.include_bias is True:
            n_out += 1

        self.n_output_features = n_out

        return self


    def transform(self, X):
        """
        Gera a matriz de características polinomiais a partir de X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matriz com os dados que serão transformados.

        Returns 
        -------
        X_poly : ndarray, shape (n_samples, n_output_features)
            Matriz com as características polinomiais.

        Raises
        ------
        ValueError
            Se X não for uma matriz 2D.
            Se o modelo não tiver sido treinado.
            Se o número de features de X for diferente.
        """
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        if not hasattr(self, "combinations_"):
            raise ValueError("O modelo não foi treinado")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("O número de variáveis deve ser igual ao usado em fit")

        n_samples, n_features = X.shape
        columns = []

        if self.include_bias is True:
            columns.append(np.ones(n_samples))

        for i in self.combinations_:
            column = np.prod(X[:, i], axis = 1)
            columns.append(column)

        X_poly = np.column_stack(columns)

        return X_poly

    def fit_transform(self, X, y = None):
        """
        Ajusta o estimador e realiza a transformação da matriz de entrada.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matriz com os dados a serem ajustados e treinados.

        y : array-like, shape (n_sample,)
            Não utilizado. Mantido por padrão da interface.

        Returns
        -------
        X_poly : ndarray, shape (n_samples, n_output_features)
            Matriz com as características polinomiais.
        """
        X_poly = self.fit(X).transform(X)
        return X_poly