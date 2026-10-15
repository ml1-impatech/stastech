import numpy as np
import pytest

from statstech.utils.validation import (NotFittedError, 
                                        check_array)

# testes de check_array

def test_check_array_converte_lista_para_array():
    # testa se a conversão para array está ocorrendo
    X = check_array([[1, 2], [3, 4]])
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)

def test_check_array_1D():
    # testa o caso em que X é um array 1D
    with pytest.raises(ValueError, match="reshape"):
        check_array([1, 2, 3])

@pytest.mark.parametrize("X", [5, np.zeros((2, 2, 2))])
def test_check_array_nao_2d(X):
    # testa o caso X escalar e X array 3D
    with pytest.raises(ValueError, match="não é 2D"):
        check_array(X)

@pytest.mark.parametrize("X", [np.empty((0,3)), np.empty((3, 0))])
def test_check_array_vazio(X):
    # testa o caso em que X é um array vazio
    with pytest.raises(ValueError, match="vazio"):
        check_array(X)

@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_check_array_nan_ou_infinito(valor):
    # testa o caso em que X contém NaN ou infinito, com alow_nan=False
    with pytest.raises(ValueError, match="NaN ou valores infinitos"):
        check_array([[1.0, valor], [3.0, 4.0]])

def test_check_array_aceita_nan():
    # testa o caso em que X contém NaN, com alow_nan=True
    X = check_array([[1.0, np.nan], [3.0, 4.0]], allow_nan=True)
    assert np.isnan(X[0, 1])
 
def test_check_array_allow_nan_proibe_infinito():
    #testa o caso em que X contém infinito, com alow_nan=True
    with pytest.raises(ValueError, match="infinitos"):
        check_array([[1.0, np.inf], [3.0, 4.0]], allow_nan=True)

def test_check_array_texto_proibido():
    # testa o caso X com texto e ensure_numeric=True
    with pytest.raises(ValueError, match="numérico"):
        check_array([["a", "b"], ["c", "d"]])

def test_check_array_sem_ensure_numeric():
    # testa o caso X com texto e ensure_numeric=False
    X = check_array([["a", "b"], ["c", "d"]], ensure_numeric=False)
    assert X.shape == (2, 2)
