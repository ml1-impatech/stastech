import numpy as np
import pytest
from statstech.decomposition.pca import PCA
from statstech.utils.validation import NotFittedError

RNG = np.random.default_rng(0)
X = RNG.normal(size=(100, 4)) * [5.0, 3.0, 1.0, 0.5]

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

def test_dados_numa_reta():
    t = np.linspace(0, 1, 30)
    pca = PCA().fit(np.column_stack([t, 2 * t]))
    np.testing.assert_allclose(pca.explained_variance_ratio_[0], 1.0)

def test_mean():
    np.testing.assert_allclose(PCA().fit(X).mean_, np.mean(X, axis=0))

def test_variancia_igual_autovalores_da_covariancia():
    autovalores = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    np.testing.assert_allclose(PCA().fit(X).explained_variance_,
                               autovalores)

@pytest.mark.parametrize("formato", [(100, 4), (3, 5)])
def test_n_components_none_usa_minimo(formato):
    X_ = RNG.normal(size=formato)
    assert PCA().fit(X_).n_components_ == min(formato)

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_n_components_inteiro_formatos(k):
    pca = PCA(n_components=k).fit(X)
    assert pca.components_.shape == (k, 4)
    assert pca.transform(X).shape == (100, k)

def test_n_components_fracao():
    todas = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    k = PCA(n_components=0.9).fit(X).n_components_
    # k é o menor número de componentes que chega a 0.9.
    assert todas[k - 1] >= 0.9
    assert k == 1 or todas[k - 2] < 0.9

def test_get_params_nao_muda_depois_do_fit():
    pca = PCA(n_components=0.9).fit(X)
    assert pca.get_params()["n_components"] == 0.9

@pytest.mark.parametrize("n_components", [0, -1, 5, 1.5, True])
def test_n_components_invalido(n_components):
    with pytest.raises(ValueError, match="n_components"):
        PCA(n_components=n_components).fit(X)
 
def test_transform_antes_do_fit():
    with pytest.raises(NotFittedError):
        PCA().transform(X)
 
def test_numero_de_colunas_errado():
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="features"):
        pca.transform(np.ones((1, 3)))
    with pytest.raises(ValueError, match="componentes"):
        pca.inverse_transform(np.ones((1, 3)))
 
 
@pytest.mark.parametrize(
    "X_ruim, mensagem",
    [
        ([[1.0, 2.0, 3.0]], "2 amostras"),
        (np.ones((5, 3)), "variância zero"),
    ],
)
def test_fit_X_invalido(X_ruim, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        PCA().fit(X_ruim)