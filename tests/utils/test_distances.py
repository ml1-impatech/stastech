"""Testes de pairwise_distances."""

import numpy as np
import pytest

from statstech.utils.distances import pairwise_distances


def test_3_4_5_triangle():
    d = pairwise_distances([[0.0, 0.0]], [[3.0, 4.0]])
    np.testing.assert_allclose(d, [[5.0]])


def test_output_shape():
    A = np.zeros((4, 3))
    B = np.ones((6, 3))
    assert pairwise_distances(A, B).shape == (4, 6)


def test_b_none_uses_a_and_zero_diagonal():
    A = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 2.0]])
    d = pairwise_distances(A)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-7)
    np.testing.assert_allclose(d, d.T)


def test_matches_double_loop_computation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    B = rng.normal(size=(7, 3))
    expected = np.array(
        [[np.sqrt(np.sum((a - b) ** 2)) for b in B] for a in A]
    )
    np.testing.assert_allclose(pairwise_distances(A, B), expected)


def test_squared_is_the_square_of_the_distance():
    A = np.array([[0.0, 0.0], [1.0, 2.0]])
    B = np.array([[3.0, 4.0]])
    d = pairwise_distances(A, B)
    d2 = pairwise_distances(A, B, squared=True)
    np.testing.assert_allclose(d2, d**2)


def test_manhattan():
    d = pairwise_distances([[0.0, 0.0]], [[3.0, -4.0]], metric="manhattan")
    np.testing.assert_allclose(d, [[7.0]])


def test_never_negative_nor_nan():
    # Pontos quase iguais e grandes: o arredondamento pode dar < 0.
    A = np.array([[1e8 + 0.1, 1e8], [1e8, 1e8 + 0.1]])
    d = pairwise_distances(A, A)
    assert np.all(d >= 0)
    assert not np.isnan(d).any()


def test_different_number_of_features():
    with pytest.raises(ValueError, match="mesmo número de features"):
        pairwise_distances(np.zeros((2, 3)), np.zeros((2, 2)))


def test_invalid_metric():
    with pytest.raises(ValueError, match="Métrica"):
        pairwise_distances([[0.0]], [[1.0]], metric="coseno")


def test_squared_with_manhattan_raises_error():
    with pytest.raises(ValueError, match="squared"):
        pairwise_distances(
            [[0.0]], [[1.0]], metric="manhattan", squared=True
        )


def test_1d_input_raises_error():
    with pytest.raises(ValueError, match="2D"):
        pairwise_distances([1.0, 2.0], [[1.0]])
