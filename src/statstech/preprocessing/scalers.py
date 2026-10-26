"""Escalonadores: StandardScaler e MinMaxScaler."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
)


class StandardScaler(BaseEstimator, TransformerMixin):
    """Padroniza cada coluna: z = (x - média) / desvio padrão.

    A média e o desvio são calculados só no fit (use os dados de
    treino) e reaproveitados no transform, para não vazar informação
    do teste. O desvio usado é o populacional (divisão por n).

    Parameters
    ----------
    with_mean : bool, default=True
        Se True, subtrai a média de cada coluna.
    with_std : bool, default=True
        Se True, divide cada coluna pelo desvio padrão.
    columns : list de int ou None, default=None
        Índices das colunas a padronizar. As demais são devolvidas sem
        mudança. Se None, todas as colunas são padronizadas.


    Attributes
    ----------
    mean_ : ndarray de formato (n_features,)
        Média de cada coluna no treino. Nas colunas fora de columns,
        vale 0.
    scale_ : ndarray de formato (n_features,)
        Desvio padrão de cada coluna. Colunas constantes (desvio 0) e
        colunas fora de columns recebem 1.
    n_features_in_ : int
        Número de colunas vistas no fit.
    """

    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True,
                 columns: list[int] | None = None) -> None:
        """Guarda os hiperparâmetros."""
        self.with_mean = with_mean
        self.with_std = with_std
        self.columns = columns


    def _column_mask(self, n_features: int) -> NDArray[np.bool_]:
        """Máscara booleana das colunas a padronizar."""
        if self.columns is None:
            return np.ones(n_features, dtype=bool)
        columns = np.asarray(self.columns)
        if columns.ndim != 1 or columns.size == 0:
            raise ValueError("columns deve ser None ou uma lista não vazia de índices.")
        if columns.dtype == bool or not np.issubdtype(columns.dtype, np.integer):
            raise ValueError("columns deve conter só índices inteiros.")
        if np.any(columns < 0) or np.any(columns >= n_features):
            raise ValueError(f"columns deve ter índices entre 0 e {n_features - 1}.")
        if len(np.unique(columns)) != len(columns):
            raise ValueError("columns não pode ter índices repetidos.")
        mask = np.zeros(n_features, dtype=bool)
        mask[columns] = True
        return mask

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        """Calcula a média e o desvio padrão de cada coluna.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : None
            Ignorado (existe só por compatibilidade com os outros
            transformadores).

        Returns
        -------
        self : StandardScaler
            O próprio objeto, ajustado.

        Raises
        ------
        ValueError
            Se X for inválido (não 2D, vazio, não numérico, com NaN
            ou infinito).
            Se columns não for None nem uma lista não vazia de índices
            inteiros, distintos e entre 0 e n_features - 1.
        """
        X = check_array(X).astype(float)
        mask = self._column_mask(X.shape[1])

        self.mean_ = np.where(mask, X.mean(axis=0), 0.0)
        std = X.std(axis=0)
        self.scale_ = np.where(mask & (std != 0), std, 1.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Aplica a padronização com a média e o desvio do fit.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados a transformar.

        Returns
        -------
        X_new : ndarray de formato (n_amostras, n_features)
            Dados padronizados.

        Raises
        ------
        NotFittedError
            Se o escalonador ainda não foi ajustado.
        ValueError
            Se X for inválido ou tiver número de colunas diferente
            do fit.
        """
        check_is_fitted(self, ["mean_", "scale_", "n_features_in_"])
        X = check_array(X).astype(float)
        check_n_features(X, self.n_features_in_)

        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Desfaz a padronização (volta à escala original).

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados padronizados.

        Returns
        -------
        X_orig : ndarray de formato (n_amostras, n_features)
            Dados na escala original.

        Raises
        ------
        NotFittedError
            Se o escalonador ainda não foi ajustado.
        ValueError
            Se X for inválido ou tiver número de colunas diferente
            do fit.
        """
        check_is_fitted(self, ["mean_", "scale_", "n_features_in_"])
        X = check_array(X).astype(float)
        check_n_features(X, self.n_features_in_)

        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Leva cada coluna para um intervalo, por padrão [0, 1].

    z = (x - mín) / (máx - mín), depois ajustado ao intervalo pedido.
    O mínimo e o máximo vêm do fit; no transform, valores fora dessa
    faixa ficam fora do intervalo de saída (não são cortados).

    Parameters
    ----------
    feature_range : tuple de dois números, default=(0, 1)
        Intervalo de saída (mínimo, máximo). O mínimo deve ser menor
        que o máximo.

    Attributes
    ----------
    data_min_ : ndarray de formato (n_features,)
        Mínimo de cada coluna no treino.
    data_max_ : ndarray de formato (n_features,)
        Máximo de cada coluna no treino.
    data_range_ : ndarray de formato (n_features,)
        Máximo menos mínimo. Colunas constantes (amplitude 0)
        recebem 1, para evitar divisão por zero.
    n_features_in_ : int
        Número de colunas vistas no fit.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        """Guarda os hiperparâmetros."""
        self.feature_range = feature_range

    def _get_range(self) -> tuple[float, float]:
        """Valida feature_range e devolve (mínimo, máximo)."""
        try:
            low, high = self.feature_range
        except (TypeError, ValueError):
            raise ValueError(
                "feature_range deve ser uma tupla (mínimo, máximo)."
            ) from None
        if not low < high:
            raise ValueError("feature_range: o mínimo deve ser menor que o máximo.")
        return float(low), float(high)

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        """Calcula o mínimo e o máximo de cada coluna.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : None
            Ignorado.

        Returns
        -------
        self : MinMaxScaler
            O próprio objeto, ajustado.

        Raises
        ------
        ValueError
            Se X for inválido ou se feature_range for inválido.
        """
        self._get_range()
        X = check_array(X).astype(float)

        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        amplitude = self.data_max_ - self.data_min_
        self.data_range_ = np.where(amplitude == 0, 1.0, amplitude)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Aplica a escala com o mínimo e o máximo do fit.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados a transformar.

        Returns
        -------
        X_new : ndarray de formato (n_amostras, n_features)
            Dados no intervalo feature_range.

        Raises
        ------
        NotFittedError
            Se o escalonador ainda não foi ajustado.
        ValueError
            Se X for inválido ou tiver número de colunas diferente
            do fit.
        """
        check_is_fitted(self, ["data_min_", "data_range_", "n_features_in_"])
        low, high = self._get_range()
        X = check_array(X).astype(float)
        check_n_features(X, self.n_features_in_)

        X_01 = (X - self.data_min_) / self.data_range_
        return X_01 * (high - low) + low

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Desfaz a escala (volta aos valores originais).

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados na escala de feature_range.

        Returns
        -------
        X_orig : ndarray de formato (n_amostras, n_features)
            Dados na escala original.

        Raises
        ------
        NotFittedError
            Se o escalonador ainda não foi ajustado.
        ValueError
            Se X for inválido ou tiver número de colunas diferente
            do fit.
        """
        check_is_fitted(self, ["data_min_", "data_range_", "n_features_in_"])
        low, high = self._get_range()
        X = check_array(X).astype(float)
        check_n_features(X, self.n_features_in_)

        X_01 = (X - low) / (high - low)
        return X_01 * self.data_range_ + self.data_min_
