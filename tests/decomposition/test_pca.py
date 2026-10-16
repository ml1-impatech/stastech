import numpy as np
import pytest
from statstech.decomposition.pca import PCA
from statstech.utils.validation import NotFittedError
from statstech.base import clone

X = np.random.default_rng(0).normal(size=(100, 4)) * [5.0, 3.0, 1.0, 0.5]

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
    x = np.linspace(0, 1, 30)
    pca = PCA().fit(np.column_stack([x, 2 * x]))
    np.testing.assert_allclose(pca.explained_variance_ratio_[0], 1.0)
    np.testing.assert_allclose(pca.components_[0],
                               np.array([1.0, 2.0]) / np.sqrt(5))

def test_mean():
    np.testing.assert_allclose(PCA().fit(X).mean_, np.mean(X, axis=0))

def test_variancia_igual_autovalores_da_covariancia():
    autovalores = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    np.testing.assert_allclose(PCA().fit(X).explained_variance_,
                               autovalores)

@pytest.mark.parametrize("formato", [(100, 4), (3, 5)])
def test_n_components_none_usa_minimo(formato):
    X_ = np.random.default_rng(0).normal(size=formato)
    assert PCA().fit(X_).n_components_ == min(formato)

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_n_components_inteiro_formatos(k):
    pca = PCA(n_components=k).fit(X)
    assert pca.components_.shape == (k, 4)
    assert pca.transform(X).shape == (100, k)

def test_n_components_fracao():
    todas = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    k = PCA(n_components=0.9).fit(X).n_components_
    assert todas[k - 1] >= 0.9
    assert k == 1 or todas[k - 2] < 0.9

def test_get_params_nao_muda_depois_do_fit():
    pca = PCA(n_components=0.9).fit(X)
    assert pca.get_params()["n_components"] == 0.9

@pytest.mark.parametrize("n_components", [0, -1, 5, 1.5, True])
def test_n_components_invalido(n_components):
    with pytest.raises(ValueError, match="n_components"):
        PCA(n_components=n_components).fit(X)
 
@pytest.mark.parametrize("metodo", ["transform", "inverse_transform"])
def test_antes_do_fit(metodo):
    with pytest.raises(NotFittedError):
        getattr(PCA(), metodo)(X)
 
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

def test_fit_transform_igual_a_fit_depois_transform():
    np.testing.assert_allclose(PCA(n_components=2).fit_transform(X),
                               PCA(n_components=2).fit(X).transform(X))

def test_transform_usa_media_do_treino():
    X_novo = np.random.default_rng(1).normal(size=(10, 4)) + 3.0
    pca = PCA(n_components=2).fit(X)
    esperado = (X_novo - pca.mean_) @ pca.components_.T
    np.testing.assert_allclose(pca.transform(X_novo), esperado)

def test_fit_devolve_self():
    pca = PCA()
    assert pca.fit(X) is pca

def test_clone_devolve_pca_nao_treinado():
    novo = clone(PCA(n_components=2).fit(X))
    assert isinstance(novo, PCA)
    assert novo.n_components == 2
    assert not hasattr(novo, "components_")