import numpy as np
import pytest
from statstech.base import clone
from statstech.cluster.kmeans import KMeans
from statstech.utils.validation import NotFittedError
 
CENTERS = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
GROUPS = np.repeat([0, 1, 2], 30)
X = CENTERS[GROUPS] + np.random.default_rng(0).normal(scale=0.3,
                                                      size=(90, 2))
 
@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_finds_separated_groups(init):
    labels = KMeans(n_clusters=3, init=init, random_state=0).fit(X).labels_
    found = [np.unique(labels[GROUPS == g]) for g in range(3)]
    assert all(len(f) == 1 for f in found)
    assert len({f[0] for f in found}) == 3
 
def test_inertia_never_increases():
    km = KMeans(n_clusters=3)
    start = km._init_random(X, np.random.default_rng(1))
    inertias = []
    for m in range(1, 8):
        km.max_iter = m
        inertias.append(km._lloyd(X, start)[3])
    assert np.all(np.diff(inertias) <= 1e-12)
 
def test_same_random_state_same_result():
    a = KMeans(n_clusters=3, random_state=42).fit(X)
    b = KMeans(n_clusters=3, random_state=42).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.labels_, b.labels_)
 
def test_one_cluster_per_point_gives_zero_inertia():
    X_small = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [5.0, 5.0]])
    km = KMeans(n_clusters=4, random_state=0).fit(X_small)
    np.testing.assert_allclose(km.inertia_, 0.0, atol=1e-12)
 
def test_attributes_consistent():
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.cluster_centers_.shape == (3, 2)
    assert km.labels_.shape == (90,)
    expected = np.sum((X - km.cluster_centers_[km.labels_]) ** 2)
    np.testing.assert_allclose(km.inertia_, expected)
 
def test_centers_are_means_of_their_points():
    km = KMeans(n_clusters=3, tol=0, random_state=0).fit(X)
    for j in range(3):
        np.testing.assert_allclose(km.cluster_centers_[j],
                                   X[km.labels_ == j].mean(axis=0))
 
def test_kmeanspp_picks_far_point():
    X_far = np.vstack([np.random.default_rng(0).normal(size=(99, 2)),
                       [[1000.0, 1000.0]]])
    km = KMeans(n_clusters=2)
    for seed in range(10):
        start = km._init_kmeanspp(X_far, np.random.default_rng(seed))
        assert start.max() > 500
 
def test_predict_and_fit_predict_match_labels():
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(km.predict(X), km.labels_)
    np.testing.assert_array_equal(
        KMeans(n_clusters=3, random_state=0).fit_predict(X), km.labels_)
 
def test_empty_cluster_does_not_produce_nan():
    km = KMeans(n_clusters=3, max_iter=10)
    start = np.array([[0.0, 0.0], [5.0, 5.0], [100.0, 100.0]])
    labels, centers, _, _ = km._lloyd(X, start)
    assert not np.isnan(centers).any()
    assert len(np.unique(labels)) == 3

@pytest.mark.parametrize(
    "params",
    [
        {"n_clusters": 0},
        {"n_clusters": 91},
        {"n_clusters": 2.0},
        {"n_clusters": True},
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": -1},
        {"init": "aleatorio"},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        KMeans(**params).fit(X)
 
def test_predict_before_fit():
    with pytest.raises(NotFittedError):
        KMeans().predict(X)
 
def test_predict_wrong_number_of_features():
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    with pytest.raises(ValueError, match="features"):
        km.predict(np.ones((1, 3)))