"""Escaladores de variáveis."""

import numpy as np

from statstech.base import BaseEstimator, TransformerMixin


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
        # 1. validar X (2D, não vazio, sem NaN) e converter para float
        # 2. calcular média e desvio de cada coluna (axis=0)
        # 3. trocar desvio zero por 1
        # 4. guardar mean_, scale_ e n_features_in_
        # 5. devolver self
        raise NotImplementedError

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
        # 1. conferir que o fit foi feito
        # 2. validar X e conferir o número de colunas
        # 3. aplicar (X - mean_) / scale_, respeitando with_mean e with_std
        # 4. devolver o array novo
        raise NotImplementedError

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
        # a conta ao contrário: X * scale_ + mean_ (respeitando os parâmetros)
        raise NotImplementedError