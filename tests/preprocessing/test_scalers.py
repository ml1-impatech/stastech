"""Testes de StandardScaler e MinMaxScaler."""

import numpy as np
import pytest

from statstech.preprocessing.scalers import MinMaxScaler, StandardScaler
from statstech.utils.validation import NotFittedError


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 4.0, 30.0],
                      size=(50, 3))


# ---------------- StandardScaler ----------------


def test_standard_media_0_e_desvio_1(X):
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0)


def test_standard_exemplo_feito_a_mao():
    # média 2, desvio populacional 1 -> z = -1, 0, 1 (para 1, 2, 3)
    # desvio = sqrt(((1-2)^2 + 0 + (3-2)^2)/3) = sqrt(2/3)
    scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(scaler.mean_, [2.0])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(2 / 3)])
    np.testing.assert_allclose(
        scaler.transform([[2.0]]), [[0.0]], atol=1e-12
    )


def test_standard_inverse_transform(X):
    scaler = StandardScaler().fit(X)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(X)), X
    )


def test_standard_usa_estatisticas_do_treino_no_teste():
    treino = np.array([[0.0], [10.0]])  # média 5, desvio 5
    teste = np.array([[20.0]])
    scaler = StandardScaler().fit(treino)
    # (20 - 5) / 5 = 3, e não 0 (que seria se recalculasse no teste)
    np.testing.assert_allclose(scaler.transform(teste), [[3.0]])


def test_standard_coluna_constante_nao_gera_nan():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    Z = StandardScaler().fit_transform(X)
    assert not np.isnan(Z).any()
    np.testing.assert_allclose(Z[:, 1], 0.0)


def test_standard_with_mean_e_with_std_false():
    X = np.array([[1.0], [3.0]])  # média 2, desvio 1
    np.testing.assert_allclose(
        StandardScaler(with_mean=False).fit_transform(X), [[1.0], [3.0]]
    )
    np.testing.assert_allclose(
        StandardScaler(with_std=False).fit_transform(X), [[-1.0], [1.0]]
    )


def test_standard_nao_altera_o_x_original(X):
    copia = X.copy()
    StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(X, copia)


def test_standard_atributos(X):
    scaler = StandardScaler().fit(X)
    assert scaler.mean_.shape == (3,)
    assert scaler.scale_.shape == (3,)
    assert scaler.n_features_in_ == 3


def test_standard_transform_antes_do_fit():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.zeros((2, 2)))


def test_standard_numero_de_colunas_diferente(X):
    scaler = StandardScaler().fit(X)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.zeros((2, 5)))


def test_standard_entrada_invalida():
    with pytest.raises(ValueError, match="2D"):
        StandardScaler().fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        StandardScaler().fit([[1.0], [np.nan]])


# ---------------- MinMaxScaler ----------------


def test_minmax_vai_de_0_a_1(X):
    Z = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0)
    np.testing.assert_allclose(Z.max(axis=0), 1.0)


def test_minmax_exemplo_feito_a_mao():
    scaler = MinMaxScaler().fit([[0.0], [5.0], [10.0]])
    np.testing.assert_allclose(scaler.data_min_, [0.0])
    np.testing.assert_allclose(scaler.data_max_, [10.0])
    np.testing.assert_allclose(
        scaler.transform([[2.5]]), [[0.25]]
    )


def test_minmax_feature_range_personalizado():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Z = scaler.fit_transform([[0.0], [5.0], [10.0]])
    np.testing.assert_allclose(Z.ravel(), [-1.0, 0.0, 1.0])


def test_minmax_inverse_transform(X):
    scaler = MinMaxScaler(feature_range=(2, 5)).fit(X)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(X)), X
    )


def test_minmax_teste_fora_da_faixa_do_treino_nao_e_cortado():
    scaler = MinMaxScaler().fit([[0.0], [10.0]])
    np.testing.assert_allclose(
        scaler.transform([[20.0], [-10.0]]), [[2.0], [-1.0]]
    )


def test_minmax_coluna_constante_nao_gera_nan():
    Z = MinMaxScaler().fit_transform([[3.0], [3.0], [3.0]])
    assert not np.isnan(Z).any()


def test_minmax_feature_range_invalido():
    with pytest.raises(ValueError, match="mínimo deve ser menor"):
        MinMaxScaler(feature_range=(1, 1)).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="tupla"):
        MinMaxScaler(feature_range=5).fit([[0.0], [1.0]])


def test_minmax_antes_do_fit_e_colunas_erradas(X):
    with pytest.raises(NotFittedError):
        MinMaxScaler().transform(X)
    scaler = MinMaxScaler().fit(X)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.zeros((2, 1)))
