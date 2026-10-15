import numpy as np
import pytest

from statstech.utils.validation import (NotFittedError, 
                                        check_array,
                                        check_X_y)

# testes de check_array

def test_check_array_converte_lista_para_array():
    X = check_array([[1, 2], [3, 4]])
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)

def test_check_array_1D():
    with pytest.raises(ValueError, match="reshape"):
        check_array([1, 2, 3])

@pytest.mark.parametrize("X", [5, np.zeros((2, 2, 2))])
def test_check_array_nao_2d(X):
    with pytest.raises(ValueError, match="não é 2D"):
        check_array(X)

@pytest.mark.parametrize("X", [np.empty((0,3)), np.empty((3, 0))])
def test_check_array_vazio(X):
    with pytest.raises(ValueError, match="vazio"):
        check_array(X)

@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_check_array_nan_ou_infinito(valor):
    with pytest.raises(ValueError, match="NaN ou valores infinitos"):
        check_array([[1.0, valor], [3.0, 4.0]])

def test_check_array_aceita_nan():
    X = check_array([[1.0, np.nan], [3.0, 4.0]], allow_nan=True)
    assert np.isnan(X[0, 1])
 
def test_check_array_allow_nan_proibe_infinito():
    with pytest.raises(ValueError, match="infinitos"):
        check_array([[1.0, np.inf], [3.0, 4.0]], allow_nan=True)

def test_check_array_texto_proibido():
    with pytest.raises(ValueError, match="numérico"):
        check_array([["a", "b"], ["c", "d"]])

def test_check_array_sem_ensure_numeric():
    X = check_array([["a", "b"], ["c", "d"]], ensure_numeric=False)
    assert X.shape == (2, 2)


# testes de check_X_y

X_ok = [[1, 2], [3, 4], [5, 6]]

def test_check_X_y_y_numerico():
    X, y = check_X_y(X_ok, [1.0, 2.0, 3.0])
    assert X.shape == (3, 2)
    assert y.shape == (3,)
 
 
def test_check_X_y_y_texto():
    _, y = check_X_y(X_ok, ["sim", "não", "sim"])
    assert list(y) == ["sim", "não", "sim"]

def test_check_X_y_tamanhos_diferentes():
    with pytest.raises(ValueError, match="3 e 2"):
        check_X_y(X_ok, [1, 2])

def test_check_X_y_sugere_ravel():
    with pytest.raises(ValueError, match="ravel"):
        check_X_y(X_ok, [[1], [2], [3]])

@pytest.mark.parametrize("y", [5, np.zeros((3, 2))])
def test_check_X_y_y_nao_1d(y):
    with pytest.raises(ValueError, match="1D"):
        check_X_y(X_ok, y)

@pytest.mark.parametrize("valor", [np.nan, np.inf])
def test_check_X_y_y_numerico_com_nan_ou_infinito(valor):
    with pytest.raises(ValueError, match="NaN ou valores infinitos"):
        check_X_y(X_ok, [1.0, valor, 3.0])