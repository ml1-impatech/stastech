"Transforma características das variáveis em polinomiais"

import itertools

import numpy as np

from statstech.base import BaseEstimator
from statstech.utils.validation import check_array, check_is_fitted, check_n_features


class PolynomialFeatures(BaseEstimator):
    """
    Transforma as variáveis de entrada em características polinomiais.

    Gera produtos das variáveis de entrada até o grau máximo especificado,
    permitindo apresentar relações polinomiais entre as variáveis.

    Parameters
    ----------
    degree: int, default=2
        Grau máximo do polinômio gerado

    include_bias: bool, default=False
        Se True, adiciona a coluna do intercepto
        preenchida com uns na matriz.
    """

    def __init__(self, degree: int = 2, include_bias: bool = False) -> None:

        self.degree = degree
        self.include_bias = include_bias

    def check_params(self):
        """
        Valida os hiperparâmetros do modelo.

        Raises
        ------
        ValueError
            Se degree não for um valor inteiro positivo.
            Se include_bias não for booleano.
        """

        if (
            not isinstance(self.degree, (int, np.integer))
            or isinstance(self.degree, (bool, np.bool_))
            or self.degree < 1
        ):
            raise ValueError("degree deve ser um valor inteiro positivo")

        if not isinstance(self.include_bias, (bool, np.bool_)):
            raise ValueError("include_bias deve ser booleano")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PolynomialFeatures":
        """
        Calcula as combinações de características polinomiais
        com base nos dados de treino.

        Parameters
        -----------
        X: array-like de formato (n_samples, n_features)
            Dados de treinamento utilizados para realizar o cálculo das combinações.

        y: array-like de formato (n_samples,)
            Não utilizado. Mantido por padrão de interface.

        Returns
        -------
        self: object
            Retorna a instância do próprio estimador.

        Raises
        ------
        ValueError
            Se X não for uma matriz bidimensional
            Se X estiver vazio
        """

        self.check_params()

        X = check_array(X)

        self.n_features_in_ = X.shape[1]

        self.combinations_ = []
        for d in range(1, self.degree + 1):
            for i in itertools.combinations_with_replacement(
                range(self.n_features_in_), d
            ):
                self.combinations_.append(i)

        n_out = len(self.combinations_)

        if self.include_bias:
            n_out += 1

        self.n_output_features_ = n_out

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Gera a matriz de características polinomiais a partir de X.

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Matriz com os dados que serão transformados.

        Returns
        -------
        X_poly : ndarray de formato (n_samples, n_output_features_)
            Matriz com as características polinomiais.

        Raises
        ------
        ValueError
            Se X não for uma matriz bidimensional.
            Se o modelo não tiver sido treinado.
            Se o número de features de X for diferente.
        """
        check_is_fitted(self, ["combinations_", "n_features_in_", "n_output_features_"])

        X = check_array(X)

        check_n_features(X, self.n_features_in_)

        n_samples = X.shape[0]
        columns = []

        if self.include_bias:
            columns.append(np.ones(n_samples))

        for i in self.combinations_:
            column = np.prod(X[:, i], axis=1)
            columns.append(column)

        X_poly = np.column_stack(columns)

        return X_poly

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Ajusta o estimador e realiza a transformação da matriz de entrada.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matriz com os dados a serem ajustados e treinados.

        y : array-like, shape (n_samples,)
            Não utilizado. Mantido por padrão da interface.

        Returns
        -------
        X_poly : ndarray, shape (n_samples, n_output_features_)
            Matriz com as características polinomiais.
        """
        X_poly = self.fit(X, y).transform(X)
        return X_poly
