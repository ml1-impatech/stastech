"""Preenchimento de valores faltantes (NaN)."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
)

_STRATEGIES = ("mean", "median", "most_frequent")


def _most_frequent(column: NDArray[np.float64]) -> float:
    """Valor mais comum da coluna (empate: o menor valor)."""
    values, counts = np.unique(column, return_counts=True)
    return float(values[np.argmax(counts)])


class SimpleImputer(BaseEstimator, TransformerMixin):
    """Troca NaN pela média, mediana ou valor mais comum da coluna.

    O valor de preenchimento é calculado só no fit (com os dados de
    treino) e reaproveitado no transform.

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent"}, default="mean"
        Como calcular o valor de preenchimento de cada coluna. Em
        "most_frequent", empates são resolvidos pelo menor valor.

    Attributes
    ----------
    statistics_ : ndarray de formato (n_features,)
        Valor de preenchimento de cada coluna.
    n_features_in_ : int
        Número de colunas vistas no fit.
    """

    def __init__(self, strategy: str = "mean") -> None:
        """Guarda os hiperparâmetros."""
        self.strategy = strategy

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        """Calcula o valor de preenchimento de cada coluna.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino, podem conter NaN.
        y : None
            Ignorado.

        Returns
        -------
        self : SimpleImputer
            O próprio objeto, ajustado.

        Raises
        ------
        ValueError
            Se strategy não existir, se X for inválido (valores
            infinitos, não numérico, não 2D) ou se alguma coluna
            tiver só NaN (não há valor para calcular).
        """
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy '{self.strategy}' inválida. Opções: {list(_STRATEGIES)}"
            )

        X = check_array(X, allow_nan=True).astype(float)

        all_nan = np.isnan(X).all(axis=0)
        if all_nan.any():
            cols = np.flatnonzero(all_nan).tolist()
            raise ValueError(f"As colunas {cols} só têm NaN, não dá para preenchê-las.")

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        else:
            self.statistics_ = np.array(
                [_most_frequent(col[~np.isnan(col)]) for col in X.T]
            )

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Preenche os NaN com os valores calculados no fit.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados a preencher, podem conter NaN.

        Returns
        -------
        X_new : ndarray de formato (n_amostras, n_features)
            Cópia de X sem NaN. Os valores que não eram NaN ficam
            iguais.

        Raises
        ------
        NotFittedError
            Se o imputer ainda não foi ajustado.
        ValueError
            Se X for inválido ou tiver número de colunas diferente
            do fit.
        """
        check_is_fitted(self, ["statistics_", "n_features_in_"])
        X = check_array(X, allow_nan=True).astype(float)

        check_n_features(X, self.n_features_in_)

        # np.where devolve um array novo: o X original não é alterado.
        return np.where(np.isnan(X), self.statistics_, X)
