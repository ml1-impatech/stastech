import numpy as np
import pytest
from statstech.decomposition.pca import PCA
from statstech.utils.validation import NotFittedError
from statstech.base import clone
 
X = np.random.default_rng(0).normal(size=(100, 4)) * [5.0, 3.0, 1.0, 0.5]
 
def test_components_orthonormal():
    C = PCA().fit(X).components_
    np.testing.assert_allclose(C @ C.T, np.eye(4), atol=1e-10)
 
def test_explained_variance_ratio_sums_to_1():
    pca = PCA().fit(X)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0)
 
def test_inverse_transform_reconstructs_X():
    pca = PCA().fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X)
 
def test_transform_shape_and_variance():
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (100, 2)
    np.testing.assert_allclose(Z.var(axis=0, ddof=1),
                               pca.explained_variance_)
 
def test_sign_convention():
    C = PCA().fit(X).components_
    largest = C[np.arange(4), np.argmax(np.abs(C), axis=1)]
    assert np.all(largest > 0)
 
def test_data_on_a_line():
    x = np.linspace(0, 1, 30)
    pca = PCA().fit(np.column_stack([x, 2 * x]))
    np.testing.assert_allclose(pca.explained_variance_ratio_[0], 1.0)
    np.testing.assert_allclose(pca.components_[0],
                               np.array([1.0, 2.0]) / np.sqrt(5))
 
def test_mean():
    np.testing.assert_allclose(PCA().fit(X).mean_, np.mean(X, axis=0))
 
def test_variance_equals_covariance_eigenvalues():
    eigenvalues = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    np.testing.assert_allclose(PCA().fit(X).explained_variance_,
                               eigenvalues)
 
@pytest.mark.parametrize("shape", [(100, 4), (3, 5)])
def test_n_components_none_uses_minimum(shape):
    X_ = np.random.default_rng(0).normal(size=shape)
    assert PCA().fit(X_).n_components_ == min(shape)
 
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_n_components_int_shapes(k):
    pca = PCA(n_components=k).fit(X)
    assert pca.components_.shape == (k, 4)
    assert pca.transform(X).shape == (100, k)
 
def test_n_components_fraction():
    cumulative = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    k = PCA(n_components=0.9).fit(X).n_components_
    assert cumulative[k - 1] >= 0.9
    assert k == 1 or cumulative[k - 2] < 0.9
 
def test_get_params_unchanged_after_fit():
    pca = PCA(n_components=0.9).fit(X)
    assert pca.get_params()["n_components"] == 0.9
 
@pytest.mark.parametrize("n_components", [0, -1, 5, 1.5, True])
def test_n_components_invalid(n_components):
    with pytest.raises(ValueError, match="n_components"):
        PCA(n_components=n_components).fit(X)
 
@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_before_fit(method):
    with pytest.raises(NotFittedError):
        getattr(PCA(), method)(X)
 
def test_wrong_number_of_columns():
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="features"):
        pca.transform(np.ones((1, 3)))
    with pytest.raises(ValueError, match="componentes"):
        pca.inverse_transform(np.ones((1, 3)))
 
@pytest.mark.parametrize(
    "X_bad, message",
    [
        ([[1.0, 2.0, 3.0]], "2 amostras"),
        (np.ones((5, 3)), "variância zero"),
    ],
)
def test_fit_invalid_X(X_bad, message):
    with pytest.raises(ValueError, match=message):
        PCA().fit(X_bad)
 
def test_fit_transform_equals_fit_then_transform():
    np.testing.assert_allclose(PCA(n_components=2).fit_transform(X),
                               PCA(n_components=2).fit(X).transform(X))
 
def test_transform_uses_training_mean():
    X_new = np.random.default_rng(1).normal(size=(10, 4)) + 3.0
    pca = PCA(n_components=2).fit(X)
    expected = (X_new - pca.mean_) @ pca.components_.T
    np.testing.assert_allclose(pca.transform(X_new), expected)
 
def test_fit_returns_self():
    pca = PCA()
    assert pca.fit(X) is pca
 
def test_clone_returns_unfitted_pca():
    new = clone(PCA(n_components=2).fit(X))
    assert isinstance(new, PCA)
    assert new.n_components == 2
    assert not hasattr(new, "components_")