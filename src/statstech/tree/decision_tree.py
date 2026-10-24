"""Árvores de decisão (CART) para regressão e classificação."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statstech.utils.validation import check_array, check_is_fitted, check_X_y

_T = TypeVar("_T", bound="_BaseDecisionTree")

_PURE_TOL = 1e-12


@dataclass
class _Node:
    """
    Nó da árvore.

    Um nó interno tem ``feature``, ``threshold``, ``left`` e ``right``.
    Uma folha tem apenas ``value``.
    """

    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    value: float | None = None


def _node_depth(node: _Node) -> int:
    """
    Devolve a profundidade da subárvore que começa em ``node``.

    Parameters
    ----------
    node : _Node
        Raiz da subárvore.

    Returns
    -------
    depth : int
        Número de divisões no caminho mais longo até uma folha.
        Uma folha tem profundidade 0
    """
    if node.feature is None or node.left is None or node.right is None:
        return 0
    return 1 + max(_node_depth(node.left), _node_depth(node.right))


class _BaseDecisionTree(BaseEstimator):
    """
    Classe-base das árvores de decisão

    Implementa a construção recursiva e a predição.
    As subclasses só definem o critério de impureza e o valor das folhas.

    Parameters
    ----------
    max_depth : int ou None, default=None
        Profundidade máxima da árvore.
        Se None, a árvore cresce até todos os nós ficarem puros ou pequenos
        demais para dividir.
    min_samples_split : int, default=2
        Número mínimo de amostras para tentar dividir um nó.

    Attributes
    ----------
    tree_ : _Node
        Nó raiz da árvore treinada.
    n_features_in_ : int
        Número de features vistas no "fit".
    """

    def __init__(
        self, max_depth: int | None = None, min_samples_split: int = 2
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _check_params(self) -> None:
        """
        Valida os hiperparâmetros.

        Raises
        ------
        ValueError
            Se ``max_depth`` não for None nem um inteiro >= 1,
            ou se ``min_samples_split`` não for um inteiro >= 2.
        """
        if self.max_depth is not None:
            if (
                isinstance(self.max_depth, bool)
                or not isinstance(self.max_depth, (int, np.integer))
                or self.max_depth < 1
            ):
                raise ValueError("max_depth deve ser None ou inteiro >= 1.")
        if (
            not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split deve ser inteiro >= 2.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Constrói a árvore de decisão a partir dos dados de treino.

        Parameters
        ----------
        X : array-like de shape (n_amostras, n_features)
            Dados de entrada (features).
        y : array-like de shape (n_amostras,)
            Valores (regressão) ou rótulos (classificação) de X.

        Returns
        -------
        self : objeto
            O próprio modelo treinado.

        Raises
        ------
        ValueError
            Se os hiperparâmetros forem inválidos.
            Se X não for 2D ou y não for 1D.
            Se não houver nenhuma amostra.
        """
        self._check_params()
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y = self._prepare_y(y)

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """
        Prevê o alvo de cada amostra de X.

        Parameters
        ----------
        X : array-like de shape (n_amostras, n_features)
            Dados de entrada. Precisa ter o mesmo número de features do ``fit``.

        Returns
        -------
        y_pred : ndarray de shape (n_amostras,)
            Valores previstos (regressão) ou rótulos originais (classificação).

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X não for 2D ou tiver número de features diferente do usado no ``fit``.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features, mas o modelo "
                f"foi treinado com {self.n_features_in_}."
            )

        values = []
        for x in X:
            node = self.tree_
            while node.feature is not None:
                assert node.left is not None and node.right is not None
                go_left = x[node.feature] <= node.threshold
                node = node.left if go_left else node.right
            values.append(node.value)

        return self._decode_predictions(np.array(values))

    def get_depth(self) -> int:
        """
        Devolve a profundidade da árvore treinada.

        Útil para estudar overfitting: sem ``max_depth`` a profundidade cresce
        até a árvore decorar o treino.

        Returns
        -------
        depth : int
            Número de divisões no caminho mais longo da raiz até uma folha.
            Uma árvore com só uma folha tem profundidade 0.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        """
        check_is_fitted(self, ["tree_"])
        return _node_depth(self.tree_)

    def _is_leaf(self, y: NDArray[Any], depth: int) -> bool:
        """
        Diz se um nó com alvo ``y`` e profundidade ``depth`` é folha.

        Parameters
        ----------
        y : ndarray de shape (n_amostras,)
            Alvo das amostras que chegaram ao nó.
        depth : int
            Profundidade do nó.

        Returns
        -------
        is_leaf : bool
            True se algum critério de parada for atendido.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if len(y) < self.min_samples_split:
            return True

        if self._impurity(y) <= _PURE_TOL:
            return True

        return False

    def _best_split(
        self, X: NDArray[np.float64], y: NDArray[Any]
    ) -> tuple[int | None, float | None]:
        """
        Acha a melhor divisão ``x_j <= t`` para as amostras (X, y).

        Para cada feature, ordena os valores uma única vez e calcula a
        impureza dos dois lados de todos os cortes de uma vez, com somas
        acumuladas. Custo O(n log n) por feature, em vez de O(n^2).

        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Features das amostras que chegam ao nó.
        y : ndarray de shape (n_amostras,)
            Alvo preparado por ``_prepare_y``.

        Returns
        -------
        feature : int ou None
            Índice da melhor feature. None se nenhum corte separa
            as amostras (linhas idênticas).
        threshold : float ou None
            Limiar da melhor divisão (ponto médio entre dois valores
            consecutivos distintos).
        """

        n_samples, n_features = X.shape
        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left

        best_feature, best_threshold = None, None
        best_score = np.inf

        for j in range(n_features):
            order = np.argsort(X[:, j], kind="stable")
            x_sorted = X[order, j]
            y_sorted = y[order]

            midpoints = (x_sorted[:-1] + x_sorted[1:]) / 2
            valid = midpoints < x_sorted[1:]
            if not valid.any():
                continue

            imp_left, imp_right = self._split_impurities(y_sorted, n_left, n_right)
            scores = (n_left * imp_left + n_right * imp_right) / n_samples
            scores = np.where(valid, scores, np.inf)

            k = int(np.argmin(scores))
            if scores[k] < best_score:
                best_score = scores[k]
                best_feature = j
                best_threshold = float(midpoints[k])

        return best_feature, best_threshold

    def _build(self, X: NDArray[np.float64], y: NDArray[Any], depth: int) -> _Node:
        """
        Constrói recursivamente a subárvore para as amostras (X, y).

        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Features das amostras que chegam ao nó.
        y : ndarray de shape (n_amostras,)
            Alvo preparado por ``_prepare_y``.
        depth : int
            Profundidade do nó atual.

        Returns
        -------
        node : _Node
            Folha ou nó interno com a melhor divisão encontrada.
        """

        if self._is_leaf(y, depth):
            return _Node(value=self._leaf_value(y))

        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return _Node(value=self._leaf_value(y))

        mask = X[:, feature] <= threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def _prepare_y(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError

    def _decode_predictions(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError

    def _impurity(self, y: NDArray[Any]) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: NDArray[Any]) -> float:
        raise NotImplementedError

    def _split_impurities(
        self,
        y_sorted: NDArray[Any],
        n_left: NDArray[np.intp],
        n_right: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    """Árvore de decisão (CART) para regressão.

    Cada divisão minimiza a soma ponderada da variância dos dois lados
    (equivalente a minimizar a soma dos quadrados dos resíduos). Cada
    folha prevê a média de y das amostras que chegam a ela.

    Parameters
    ----------
    max_depth : int ou None, default=None
        Profundidade máxima da árvore. Se None, a árvore cresce até
        todos os nós ficarem puros ou pequenos demais para dividir, o
        que decora o treino.
    min_samples_split : int, default=2
        Número mínimo de amostras para tentar dividir um nó.

    Attributes
    ----------
    tree_ : _Node
        Nó raiz da árvore treinada.
    n_features_in_ : int
        Número de features vistas no ``fit``.
    """

    def _prepare_y(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        """Converte y para floats."""
        return y_ou_values.astype(float)

    def _decode_predictions(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        """Devolve os valores das folhas como floats."""
        return y_ou_values.astype(float)

    def _impurity(self, y: NDArray[Any]) -> float:
        """Variância de y em torno da média (soma dos quadrados / n)."""
        return float(np.var(y))

    def _leaf_value(self, y: NDArray[Any]) -> float:
        """Média de y."""
        return float(np.mean(y))

    def _split_impurities(
        self,
        y_sorted: NDArray[Any],
        n_left: NDArray[np.intp],
        n_right: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Variância dos dois lados de cada corte, para y já ordenado.

        Usa Var = E[y^2] - E[y]^2 com somas acumuladas. Centraliza y
        antes, porque essa fórmula perde precisão quando a média é
        grande em relação ao desvio.

        Parameters
        ----------
        y_sorted : ndarray de shape (n_amostras,)
            Alvo ordenado pela feature candidata.
        n_left, n_right : ndarray de shape (n_amostras - 1,)
            Tamanho de cada lado para cada corte.

        Returns
        -------
        imp_left, imp_right : ndarray de shape (n_amostras - 1,)
            Variância do lado esquerdo e do direito de cada corte.
        """
        y_c = y_sorted - y_sorted.mean()
        sum_left = np.cumsum(y_c)[:-1]
        sq_left = np.cumsum(y_c**2)[:-1]
        sum_right = y_c.sum() - sum_left
        sq_right = (y_c**2).sum() - sq_left

        imp_left = sq_left / n_left - (sum_left / n_left) ** 2
        imp_right = sq_right / n_right - (sum_right / n_right) ** 2
        return imp_left, imp_right


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    """Árvore de decisão (CART) para classificação.

    Cada divisão minimiza a impureza de Gini ponderada dos dois lados.
    Cada folha prevê a classe majoritária; em caso de empate, a classe
    de menor índice em ``classes_``.

    Parameters
    ----------
    max_depth : int ou None, default=None
        Profundidade máxima da árvore. Se None, a árvore cresce até
        todos os nós ficarem puros ou pequenos demais para dividir, o
        que decora o treino.
    min_samples_split : int, default=2
        Número mínimo de amostras para tentar dividir um nó.

    Attributes
    ----------
    tree_ : _Node
        Nó raiz da árvore treinada.
    classes_ : ndarray de shape (n_classes,)
        Rótulos únicos de y, em ordem crescente.
    n_features_in_ : int
        Número de features vistas no ``fit``.
    """

    def _prepare_y(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        """Define ``classes_`` e troca cada rótulo pelo seu índice."""
        self.classes_, y_index = np.unique(y_ou_values, return_inverse=True)
        return y_index

    def _decode_predictions(self, y_ou_values: NDArray[Any]) -> NDArray[Any]:
        """Traduz os índices das folhas de volta para os rótulos."""
        return self.classes_[y_ou_values.astype(int)]

    def _impurity(self, y: NDArray[Any]) -> float:
        """Impureza de Gini: 1 - soma(p_k ** 2). Nó puro vale 0."""
        p = np.bincount(y) / len(y)
        return float(1 - np.sum(p**2))

    def _leaf_value(self, y: NDArray[Any]) -> float:
        """Índice da classe majoritária (empate: menor índice)."""
        return int(np.argmax(np.bincount(y)))

    def _split_impurities(
        self,
        y_sorted: NDArray[Any],
        n_left: NDArray[np.intp],
        n_right: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Gini dos dois lados de cada corte, para y já ordenado.

        Usa a contagem acumulada de cada classe (one-hot + cumsum).

        Parameters
        ----------
        y_sorted : ndarray de shape (n_amostras,)
            Índices de classe ordenados pela feature candidata.
        n_left, n_right : ndarray de shape (n_amostras - 1,)
            Tamanho de cada lado para cada corte.

        Returns
        -------
        imp_left, imp_right : ndarray de shape (n_amostras - 1,)
            Gini do lado esquerdo e do direito de cada corte.
        """
        one_hot = np.eye(len(self.classes_))[y_sorted]
        counts_left = np.cumsum(one_hot, axis=0)[:-1]
        counts_right = one_hot.sum(axis=0) - counts_left

        p_left = counts_left / n_left[:, None]
        p_right = counts_right / n_right[:, None]
        imp_left = 1 - np.sum(p_left**2, axis=1)
        imp_right = 1 - np.sum(p_right**2, axis=1)
        return imp_left, imp_right
