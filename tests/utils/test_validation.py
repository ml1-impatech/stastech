import numpy as np
import pytest

from statstech.utils.validation import (
    NotFittedError,
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)

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

def test_check_array_valores_corretos():
    X = np.array([[1.5, 2.0], [3.0, 4.5]])
    np.testing.assert_allclose(check_array(X), X)

def test_check_array_bool():
    X = check_array([[True, False], [False, True]])
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

@pytest.mark.parametrize("faltante", [None, np.nan])
def test_check_X_y_y_texto_com_nan(faltante):
    y = np.array(["a", faltante, "b"], dtype=object)
    with pytest.raises(ValueError, match="NaN ou None"):
        check_X_y(X_ok, y)

def test_check_X_y_X_invalido():
    with pytest.raises(ValueError, match="reshape"):
        check_X_y([1, 2, 3], [1, 2, 3])

def test_check_X_y_y_bool():
    _, y = check_X_y(X_ok, [True, False, True])
    assert y.shape == (3,)


# testes de check_is_fitted

class Modelo:
    """Modelo mínimo: hiperparâmetro sem _, fit cria coef_."""
 
    def __init__(self, alpha=1.0):
        self.alpha = alpha
 
    def fit(self):
        self.coef_ = np.array([1.0])
        return self
 
def test_not_fitted_error_e_value_error():
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)
 
def test_check_is_fitted_sem_lista_nao_treinado():
    with pytest.raises(NotFittedError, match="Modelo"):
        check_is_fitted(Modelo())
 
def test_check_is_fitted_sem_lista_treinado():
    check_is_fitted(Modelo().fit())
 
@pytest.mark.parametrize("attributes", ["coef_", ["coef_"]])
def test_check_is_fitted_com_lista_treinado(attributes):
    check_is_fitted(Modelo().fit(), attributes)
 
@pytest.mark.parametrize("attributes", ["coef_", ["coef_"]])
def test_check_is_fitted_com_lista_nao_treinado(attributes):
    with pytest.raises(NotFittedError, match="coef_"):
        check_is_fitted(Modelo(), attributes)
 
def test_check_is_fitted_falta_algum_a_treinar():
    with pytest.raises(NotFittedError, match=r"\['intercept_'\]"):
        check_is_fitted(Modelo().fit(), ["coef_", "intercept_"])

# testes de check_n_features

def test_check_n_features_ok():
    check_n_features(np.ones((5, 3)), 3)


@pytest.mark.parametrize("n_cols", [2, 4])
def test_check_n_features_wrong(n_cols):
    with pytest.raises(ValueError, match=f"X tem {n_cols} features"):
        check_n_features(np.ones((5, n_cols)), 3)


def test_check_n_features_uses_name():
    with pytest.raises(ValueError, match="^Z tem"):
        check_n_features(np.ones((5, 2)), 3, name="Z")
