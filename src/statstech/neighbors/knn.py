"""K vizinhos mais próximos (KNN) para classificação e regressão."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statstech.utils.distances import pairwise_distances
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)

# Número de linhas de X processadas de cada vez no cálculo das distâncias.
_BLOCK_SIZE = 256


class _BaseKNN(BaseEstimator):
    """Parte comum do KNN: guardar o treino e achar os vizinhos.

    Parameters
    ----------
    n_neighbors : int, default=5
        Número de vizinhos consultados (k).
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        """Guarda o hiperparâmetro (sem validar nada)."""
        self.n_neighbors = n_neighbors

    def _fit_base(self, X: ArrayLike, y: ArrayLike) -> None:
        """Valida os dados e guarda uma cópia do treino."""
        X, y = check_X_y(X, y)

        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("n_neighbors deve ser um número inteiro.")
        if k < 1:
            raise ValueError("n_neighbors deve ser pelo menos 1.")
        if k > X.shape[0]:
            raise ValueError(
                f"n_neighbors={k} é maior que o número de amostras "
                f"de treino ({X.shape[0]})."
            )

        self.X_train_ = X.astype(float)
        self.y_train_ = y.copy()

    def _kneighbors(self, X: ArrayLike) -> NDArray[np.intp]:
        """Índices (no treino) dos k vizinhos mais próximos de cada linha.

        Em caso de empate de distância, vence o ponto que aparece
        primeiro no conjunto de treino (a ordenação é estável).
        As distâncias são calculadas em blocos de _BLOCK_SIZE linhas de X,
        para não montar de uma vez a matriz de todas as distâncias. O
        resultado é o mesmo de calcular tudo de uma vez.

        Returns
        -------
        indices : ndarray de formato (n_amostras, n_neighbors)
            Cada linha lista os vizinhos do mais perto ao mais longe.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X)

        check_n_features(X, self.X_train_.shape[1])

        neighbors = []
        for start in range(0, X.shape[0], _BLOCK_SIZE):
            block = X[start : start + _BLOCK_SIZE]
            distances = pairwise_distances(block, self.X_train_)
            order = np.argsort(distances, axis=1, kind="stable")
            neighbors.append(order[:, : self.n_neighbors])
        return np.vstack(neighbors)


class KNeighborsClassifier(_BaseKNN, ClassifierMixin):
    """Classificador dos k vizinhos mais próximos.

    Cada ponto novo recebe a classe mais votada entre os k pontos de
    treino mais próximos (distância euclidiana). Como o KNN usa
    distâncias, recomenda-se padronizar as features antes (ver
    StandardScaler em statstech.preprocessing).

    Parameters
    ----------
    n_neighbors : int, default=5
        Número de vizinhos consultados (k).

    Attributes
    ----------
    classes_ : ndarray de formato (n_classes,)
        Rótulos originais, em ordem crescente.
    X_train_ : ndarray de formato (n_amostras, n_features)
        Cópia dos dados de treino.
    y_train_ : ndarray de formato (n_amostras,)
        Cópia dos rótulos de treino.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Guarda os dados de treino (o KNN não "aprende" nada além disso).

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : array-like de formato (n_amostras,)
            Classes de cada amostra (números ou texto).

        Returns
        -------
        self : KNeighborsClassifier
            O próprio modelo, treinado.

        Raises
        ------
        TypeError
            Se n_neighbors não for um número inteiro.
        ValueError
            Se X ou y forem inválidos, se n_neighbors for menor que 1 ou
            se for maior que o número de amostras.
        """
        self._fit_base(X, y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Proporção de cada classe entre os k vizinhos.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Pontos a classificar.

        Returns
        -------
        probabilities : ndarray de formato (n_amostras, n_classes)
            Cada linha soma 1. As colunas seguem a ordem de classes_.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X for inválido ou tiver número de features diferente
            do treino.
        """
        neighbors = self._kneighbors(X)

        # Classe de cada vizinho, como número de 0 a n_classes - 1.
        class_index = np.searchsorted(self.classes_, self.y_train_)
        neighbor_classes = class_index[neighbors]

        counts = np.zeros((neighbors.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            counts[:, c] = np.sum(neighbor_classes == c, axis=1)

        return counts / self.n_neighbors

    def predict(self, X: ArrayLike) -> NDArray:
        """Classe mais votada entre os k vizinhos.

        Em caso de empate no voto, vence a classe que vem primeiro em
        classes_ (menor rótulo).

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Pontos a classificar.

        Returns
        -------
        predictions : ndarray de formato (n_amostras,)
            Rótulos previstos, no formato original de y.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X for inválido ou tiver número de features diferente
            do treino.
        """
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class KNeighborsRegressor(_BaseKNN, RegressorMixin):
    """Regressor dos k vizinhos mais próximos.

    A previsão para um ponto novo é a média do alvo y dos k pontos de
    treino mais próximos. Recomenda-se padronizar as features antes.

    Parameters
    ----------
    n_neighbors : int, default=5
        Número de vizinhos consultados (k).

    Attributes
    ----------
    X_train_ : ndarray de formato (n_amostras, n_features)
        Cópia dos dados de treino.
    y_train_ : ndarray de formato (n_amostras,)
        Cópia do alvo de treino.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Guarda os dados de treino.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : array-like de formato (n_amostras,)
            Valores numéricos do alvo.

        Returns
        -------
        self : KNeighborsRegressor
            O próprio modelo, treinado.

        Raises
        ------
        TypeError
            Se n_neighbors não for um número inteiro.
        ValueError
            Se X ou y forem inválidos, se y não for numérico, se
            n_neighbors for menor que 1 ou se for maior que o número de
            amostras.
        """
        self._fit_base(X, y)
        if not np.issubdtype(self.y_train_.dtype, np.number):
            raise ValueError("y deve ser numérico na regressão.")
        self.y_train_ = self.y_train_.astype(float)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Média do alvo entre os k vizinhos mais próximos.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Pontos para prever.

        Returns
        -------
        predictions : ndarray de formato (n_amostras,)
            Valores previstos.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X for inválido ou tiver número de features diferente
            do treino.
        """
        neighbors = self._kneighbors(X)
        return self.y_train_[neighbors].mean(axis=1)
