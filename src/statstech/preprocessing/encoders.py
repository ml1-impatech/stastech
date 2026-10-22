"""Codificadores de categorias: LabelEncoder e OneHotEncoder."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
)


class LabelEncoder(BaseEstimator):
    """Troca rótulos (texto ou número) por inteiros de 0 a n_classes - 1.

    Serve para o alvo y, não para as features. Os inteiros seguem a
    ordem crescente (alfabética, no caso de texto) dos rótulos.

    Attributes
    ----------
    classes_ : ndarray de formato (n_classes,)
        Rótulos únicos, em ordem crescente. A posição do rótulo em
        classes_ é o seu código.
    """

    def __init__(self) -> None:
        """Não há hiperparâmetros."""

    @staticmethod
    def _check_y(y: ArrayLike) -> NDArray:
        """Converte para array e exige 1D não vazio."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y deve ser 1D, mas tem {y.ndim} dimensões.")
        if y.shape[0] == 0:
            raise ValueError("y está vazio.")
        return y

    def fit(self, y: ArrayLike) -> Self:
        """Aprende os rótulos existentes.

        Parameters
        ----------
        y : array-like de formato (n_amostras,)
            Rótulos de treino.

        Returns
        -------
        self : LabelEncoder
            O próprio objeto, ajustado.

        Raises
        ------
        ValueError
            Se y não for 1D ou estiver vazio.
        """
        y = self._check_y(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: ArrayLike) -> NDArray[np.intp]:
        """Troca cada rótulo pelo seu código.

        Parameters
        ----------
        y : array-like de formato (n_amostras,)
            Rótulos a codificar.

        Returns
        -------
        codes : ndarray de inteiros, formato (n_amostras,)
            Código de cada rótulo.

        Raises
        ------
        NotFittedError
            Se o codificador ainda não foi ajustado.
        ValueError
            Se y for inválido ou tiver rótulos que não existiam no fit.
        """
        check_is_fitted(self, ["classes_"])
        y = self._check_y(y)

        unknown = ~np.isin(y, self.classes_)
        if unknown.any():
            raise ValueError(
                f"y contém rótulos não vistos no fit: {np.unique(y[unknown]).tolist()}"
            )
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y: ArrayLike) -> NDArray[np.intp]:
        """Faz fit e transform de uma vez.

        Parameters
        ----------
        y : array-like de formato (n_amostras,)
            Rótulos.

        Returns
        -------
        codes : ndarray de inteiros, formato (n_amostras,)
            Código de cada rótulo.

        Raises
        ------
        ValueError
            Se y não for 1D ou estiver vazio.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, codes: ArrayLike) -> NDArray:
        """Volta dos códigos para os rótulos originais.

        Parameters
        ----------
        codes : array-like de inteiros, formato (n_amostras,)
            Códigos de 0 a n_classes - 1.

        Returns
        -------
        y : ndarray de formato (n_amostras,)
            Rótulos originais.

        Raises
        ------
        NotFittedError
            Se o codificador ainda não foi ajustado.
        ValueError
            Se algum código for inválido (não inteiro ou fora do
            intervalo 0 a n_classes - 1).
        """
        check_is_fitted(self, ["classes_"])
        codes = self._check_y(codes)

        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("Os códigos devem ser números inteiros.")
        if codes.min() < 0 or codes.max() >= len(self.classes_):
            raise ValueError(f"Códigos devem estar entre 0 e {len(self.classes_) - 1}.")
        return self.classes_[codes]


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Transforma colunas categóricas em colunas 0/1 (uma por categoria).

    Uma coluna com valores {A, B, C} vira três colunas: "é A?", "é B?"
    e "é C?". Cada coluna de entrada é codificada separadamente, e as
    colunas de saída aparecem na ordem das colunas de entrada.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        O que fazer com uma categoria que não existia no fit. "error"
        levanta ValueError; "ignore" gera uma linha só de zeros para
        aquela coluna.
    drop : {None, "first"}, default=None
        Se "first", remove a primeira categoria de cada coluna. Isso
        evita colinearidade nos modelos lineares (a soma das colunas
        seria sempre 1, igual ao intercepto). Com drop="first" e
        handle_unknown="ignore", uma categoria desconhecida fica
        igual à categoria removida.

    Attributes
    ----------
    categories_ : list de ndarray
        Categorias de cada coluna de entrada, em ordem crescente.
    n_features_in_ : int
        Número de colunas de entrada.
    n_features_out_ : int
        Número de colunas de saída.

    Notes
    -----
    O X de entrada deve ter um único tipo (tudo texto ou tudo
    número): converta antes se houver mistura.
    """

    def __init__(self, handle_unknown: str = "error", drop: str | None = None) -> None:
        """Guarda os hiperparâmetros."""
        self.handle_unknown = handle_unknown
        self.drop = drop

    def _check_params(self) -> None:
        """Valida os hiperparâmetros."""
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown deve ser 'error' ou 'ignore'.")
        if self.drop not in (None, "first"):
            raise ValueError("drop deve ser None ou 'first'.")

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        """Aprende as categorias de cada coluna.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados categóricos de treino (texto ou números).
        y : None
            Ignorado.

        Returns
        -------
        self : OneHotEncoder
            O próprio objeto, ajustado.

        Raises
        ------
        ValueError
            Se X for inválido ou se handle_unknown/drop forem inválidos.
        """
        self._check_params()
        X = check_array(X, ensure_numeric=False)

        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]

        n_drop = 1 if self.drop == "first" else 0
        self.n_features_out_ = sum(len(cats) - n_drop for cats in self.categories_)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Codifica X em colunas 0/1.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados categóricos a codificar.

        Returns
        -------
        X_new : ndarray de formato (n_amostras, n_features_out_)
            Matriz de 0 e 1.

        Raises
        ------
        NotFittedError
            Se o codificador ainda não foi ajustado.
        ValueError
            Se X for inválido, tiver número de colunas diferente do
            fit, ou categoria desconhecida com handle_unknown="error".
        """
        check_is_fitted(self, ["categories_", "n_features_in_"])
        self._check_params()
        X = check_array(X, ensure_numeric=False)

        check_n_features(X, self.n_features_in_)

        blocks = []
        for j, cats in enumerate(self.categories_):
            column = X[:, j]

            if self.handle_unknown == "error":
                unknown = ~np.isin(column, cats)
                if unknown.any():
                    raise ValueError(
                        f"A coluna {j} tem categorias não vistas no fit: "
                        f"{np.unique(column[unknown]).tolist()}"
                    )

            # (n_amostras, 1) == (1, n_categorias) -> (n_amostras, n_cat)
            block = (column[:, None] == cats[None, :]).astype(float)
            if self.drop == "first":
                block = block[:, 1:]
            blocks.append(block)

        return np.hstack(blocks)
