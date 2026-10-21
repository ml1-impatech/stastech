"""Testes do SimpleImputer."""

import numpy as np
import pytest

from statstech.preprocessing.imputer import SimpleImputer
from statstech.utils.validation import NotFittedError

NAN = np.nan


@pytest.fixture
def X():
    return np.array([
        [1.0, 10.0],
        [2.0, NAN],
        [NAN, 30.0],
        [5.0, 10.0],
    ])


def test_mean_preenche_com_a_media(X):
    imputer = SimpleImputer(strategy="mean").fit(X)
    np.testing.assert_allclose(imputer.statistics_, [8 / 3, 50 / 3])
    Z = imputer.transform(X)
    np.testing.assert_allclose(Z[2, 0], 8 / 3)
    np.testing.assert_allclose(Z[1, 1], 50 / 3)


def test_median_preenche_com_a_mediana(X):
    imputer = SimpleImputer(strategy="median").fit(X)
    np.testing.assert_allclose(imputer.statistics_, [2.0, 10.0])


def test_most_frequent_preenche_com_o_valor_mais_comum(X):
    imputer = SimpleImputer(strategy="most_frequent").fit(X)
    # coluna 1: 10 aparece duas vezes. Coluna 0: tudo único, menor = 1.
    np.testing.assert_allclose(imputer.statistics_, [1.0, 10.0])


def test_nao_sobra_nan(X):
    for estrategia in ("mean", "median", "most_frequent"):
        Z = SimpleImputer(strategy=estrategia).fit_transform(X)
        assert not np.isnan(Z).any()


def test_valores_que_nao_eram_nan_ficam_iguais(X):
    Z = SimpleImputer().fit_transform(X)
    presentes = ~np.isnan(X)
    np.testing.assert_array_equal(Z[presentes], X[presentes])


def test_nao_altera_o_x_original(X):
    copia = X.copy()
    SimpleImputer().fit_transform(X)
    np.testing.assert_array_equal(X, copia)  # NaN == NaN neste assert


def test_usa_valores_do_treino_no_teste():
    treino = np.array([[2.0], [4.0]])  # média 3
    teste = np.array([[NAN], [100.0]])
    imputer = SimpleImputer().fit(treino)
    np.testing.assert_allclose(imputer.transform(teste), [[3.0], [100.0]])


def test_coluna_inteira_nan_levanta_erro():
    X = np.array([[1.0, NAN], [2.0, NAN]])
    with pytest.raises(ValueError, match="só têm NaN"):
        SimpleImputer().fit(X)


def test_strategy_invalida():
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer(strategy="moda").fit([[1.0], [2.0]])


def test_infinito_levanta_erro():
    with pytest.raises(ValueError, match="infinitos"):
        SimpleImputer().fit([[1.0], [np.inf]])


def test_antes_do_fit_e_colunas_erradas(X):
    with pytest.raises(NotFittedError):
        SimpleImputer().transform(X)
    imputer = SimpleImputer().fit(X)
    with pytest.raises(ValueError, match="features"):
        imputer.transform(np.zeros((2, 3)))
