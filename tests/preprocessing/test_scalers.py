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