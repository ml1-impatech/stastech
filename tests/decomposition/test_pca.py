import numpy as np
import pytest
from statstech.decomposition.pca import PCA

def test_componentes_ortonormais():
    C = PCA().fit(X).components_
    np.testing.assert_allclose(C @ C.T, np.eye(4), atol=1e-10)
 
def test_razao_de_variancia_soma_1():
    pca = PCA().fit(X)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0)
 
def test_inverse_transform_reconstroi_X():
    pca = PCA().fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X)

def test_transform_formato_e_variancia():
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (100, 2)
    np.testing.assert_allclose(Z.var(axis=0, ddof=1),
                               pca.explained_variance_)

def test_convencao_de_sinal():
    C = PCA().fit(X).components_
    maiores = C[np.arange(4), np.argmax(np.abs(C), axis=1)]
    assert np.all(maiores > 0)