"""Testes de pairwise_distances."""

import numpy as np
import pytest

from statstech.utils.distances import pairwise_distances


def test_triangulo_3_4_5():
    d = pairwise_distances([[0.0, 0.0]], [[3.0, 4.0]])
    np.testing.assert_allclose(d, [[5.0]])


def test_formato_da_saida():
    A = np.zeros((4, 3))
    B = np.ones((6, 3))
    assert pairwise_distances(A, B).shape == (4, 6)


def test_b_none_usa_a_e_diagonal_zero():
    A = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 2.0]])
    d = pairwise_distances(A)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-7)
    np.testing.assert_allclose(d, d.T)


def test_igual_ao_calculo_com_laco_duplo():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    B = rng.normal(size=(7, 3))
    esperado = np.array(
        [[np.sqrt(np.sum((a - b) ** 2)) for b in B] for a in A]
    )
    np.testing.assert_allclose(pairwise_distances(A, B), esperado)


def test_squared_e_o_quadrado_da_distancia():
    A = np.array([[0.0, 0.0], [1.0, 2.0]])
    B = np.array([[3.0, 4.0]])
    d = pairwise_distances(A, B)
    d2 = pairwise_distances(A, B, squared=True)
    np.testing.assert_allclose(d2, d**2)


def test_manhattan():
    d = pairwise_distances([[0.0, 0.0]], [[3.0, -4.0]], metric="manhattan")
    np.testing.assert_allclose(d, [[7.0]])


def test_nunca_negativa_nem_nan():
    # Pontos quase iguais e grandes: o arredondamento pode dar < 0.
    A = np.array([[1e8 + 0.1, 1e8], [1e8, 1e8 + 0.1]])
    d = pairwise_distances(A, A)
    assert np.all(d >= 0)
    assert not np.isnan(d).any()


def test_numero_de_features_diferente():
    with pytest.raises(ValueError, match="mesmo número de features"):
        pairwise_distances(np.zeros((2, 3)), np.zeros((2, 2)))


def test_metrica_invalida():
    with pytest.raises(ValueError, match="Métrica"):
        pairwise_distances([[0.0]], [[1.0]], metric="coseno")


def test_squared_com_manhattan_levanta_erro():
    with pytest.raises(ValueError, match="squared"):
        pairwise_distances(
            [[0.0]], [[1.0]], metric="manhattan", squared=True
        )


def test_entrada_1d_levanta_erro():
    with pytest.raises(ValueError, match="2D"):
        pairwise_distances([1.0, 2.0], [[1.0]])
