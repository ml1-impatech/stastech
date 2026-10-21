"""Escaladores de variáveis."""

import numpy as np

from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import check_array, check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    """Padroniza cada coluna para média 0 e desvio padrão 1.

    A transformação é ``z = (x - média) / desvio``, calculada coluna a
    coluna. A média e o desvio são aprendidos em ``fit`` (apenas com os
    dados de treino) e reutilizados em ``transform``.

    Parameters
    ----------
    with_mean : bool, default=True
        Se True, subtrai a média de cada coluna.
    with_std : bool, default=True
        Se True, divide cada coluna pelo seu desvio padrão.

    Attributes
    ----------
    mean_ : np.ndarray de forma (n_features,)
        Média de cada coluna, calculada no ``fit``.
    scale_ : np.ndarray de forma (n_features,)
        Desvio padrão de cada coluna, calculado no ``fit``.
        Colunas constantes têm ``scale_`` igual a 1.
    n_features_in_ : int
        Número de colunas vistas no ``fit``.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: np.ndarray, y=None) -> "StandardScaler":
        """Calcula a média e o desvio padrão de cada coluna.

        Parameters
        ----------
        X : np.ndarray de forma (n_samples, n_features)
            Dados de treino.
        y : ignorado
            Existe apenas para manter a interface comum dos modelos.

        Returns
        -------
        self : StandardScaler
            O próprio objeto ajustado.

        Raises
        ------
        ValueError
            Se ``X`` não for 2D, estiver vazio ou contiver NaN.
        """
        X = check_array(X).astype(float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica a padronização usando a média e o desvio do ``fit``.

        Parameters
        ----------
        X : np.ndarray de forma (n_samples, n_features)
            Dados a transformar.

        Returns
        -------
        X_new : np.ndarray de forma (n_samples, n_features)
            Dados padronizados (um array novo, ``X`` não é alterado).

        Raises
        ------
        ValueError
            Se ``fit`` não foi chamado, ``X`` for inválido ou o número de
            colunas for diferente do visto no ``fit``.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features, mas o scaler foi ajustado "
                f"com {self.n_features_in_}."
            )
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Desfaz a padronização, voltando às unidades originais.

        Parameters
        ----------
        X : np.ndarray de forma (n_samples, n_features)
            Dados padronizados.

        Returns
        -------
        X_orig : np.ndarray de forma (n_samples, n_features)
            Dados nas unidades originais.

        Raises
        ------
        ValueError
            Se ``fit`` não foi chamado, ``X`` for inválido ou o número de
            colunas for diferente do visto no ``fit``.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features, mas o scaler foi ajustado "
                f"com {self.n_features_in_}."
            )
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X
