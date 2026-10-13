import numpy as np
import pytest

from statstech.preprocessing import StandardScaler


@pytest.fixture
def X():
    return np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])

def test_fit_calcula_media_e_desvio(X):
    scaler = StandardScaler()
    scaler.fit(X)
    np.testing.assert_allclose(scaler.mean_, [2, 200])
    np.testing.assert_allclose(scaler.scale_, np.sqrt(np.array([2/3, 20000/3])))

def test_transform_exemplo_a_mao(X):
    scaler = StandardScaler()
    scaler.fit(X)
    resultado = scaler.transform(X)
    esperado = [-1.22474487, 0, 1.22474487]
    np.testing.assert_allclose(resultado[:, 0], esperado, atol=1e-7)
    np.testing.assert_allclose(resultado[:, 1], esperado, atol=1e-7)

def test_inverse_transform_recupera_original(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, X)

def test_coluna_constante_sem_nan():
    X = np.array([[5.0, 1.0], 
                  [5.0, 2.0], 
                  [5.0, 3.0]])
    scaler = StandardScaler()
    resultado = scaler.fit_transform(X)    
    assert np.isfinite(resultado).all()
    np.testing.assert_allclose(resultado[:, 0], 0)

def test_transform_usa_estatisticas_do_treino():
    treino = np.array([[1.0], [2.0], [3.0]])
    teste = np.array([[4.0]])
    
    scaler = StandardScaler()
    scaler.fit(treino)
    
    resultado = scaler.transform(teste)
    
    np.testing.assert_allclose(resultado, [[2.44948974]])