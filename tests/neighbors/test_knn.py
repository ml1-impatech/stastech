"""Testes do KNN (classificador e regressor)."""

import numpy as np
import pytest

from statstech.neighbors.knn import KNeighborsClassifier, KNeighborsRegressor
from statstech.utils.validation import NotFittedError


@pytest.fixture
def blobs():
    """Dois grupos bem separados, 10 pontos cada."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(10, 2))
    X1 = rng.normal(loc=10.0, scale=0.5, size=(10, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 10 + [1] * 10)
    return X, y


# ---------------- Classificador ----------------


def test_hand_made_example():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 0, 1, 1])
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    # 1.4 -> vizinhos 1.0, 2.0, 0.0 (todos da classe 0)
    # 9.0 -> vizinhos 10.0, 11.0, 2.0 (dois votos para a classe 1)
    pred = model.predict(np.array([[1.4], [9.0]]))
    assert np.array_equal(pred, [0, 1])


def test_k1_is_100_percent_accurate_on_training_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, size=30)
    model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert model.score(X, y) == 1.0


def test_k_equal_to_n_samples_predicts_majority_class():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 0, 0])
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    pred = model.predict(np.array([[100.0], [-50.0], [3.5]]))
    assert np.all(pred == 1)


def test_text_labels_come_back_as_text():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["nao", "nao", "sim", "sim"])
    model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    pred = model.predict(np.array([[0.2], [10.4]]))
    assert list(pred) == ["nao", "sim"]
    assert list(model.classes_) == ["nao", "sim"]


def test_predict_proba_in_0_1_and_sums_to_1(blobs):
    X, y = blobs
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (20, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_vote_proportion():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    # vizinhos de 0.5: 0.0 (c0), 1.0 (c0), 2.0 (c1) -> [2/3, 1/3]
    proba = model.predict_proba(np.array([[0.5]]))
    np.testing.assert_allclose(proba, [[2 / 3, 1 / 3]])


def test_vote_tie_picks_smallest_class():
    X = np.array([[0.0], [2.0]])
    y = np.array([5, 3])
    model = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    # 1 voto para cada classe: vence a menor (3)
    assert model.predict(np.array([[1.0]]))[0] == 3


def test_multiclass(blobs):
    X, y = blobs
    X3 = np.vstack([X, X[:10] + 20.0])
    y3 = np.concatenate([y, np.full(10, 2)])
    model = KNeighborsClassifier(n_neighbors=3).fit(X3, y3)
    assert model.score(X3, y3) == 1.0
    assert model.predict_proba(X3).shape == (30, 3)


def test_classes_are_sorted_and_stores_training_data(blobs):
    X, y = blobs
    model = KNeighborsClassifier().fit(X, y)
    assert np.array_equal(model.classes_, [0, 1])
    assert model.X_train_.shape == X.shape
    assert model.y_train_.shape == y.shape


def test_fit_returns_self_and_copies_data(blobs):
    X, y = blobs
    model = KNeighborsClassifier()
    assert model.fit(X, y) is model
    X_original = X.copy()
    X[0, 0] = 999.0  # mexer no X do usuário não pode afetar o modelo
    np.testing.assert_allclose(model.X_train_[0], X_original[0])


def test_get_params_and_clone():
    from statstech.base import clone

    model = KNeighborsClassifier(n_neighbors=7)
    assert model.get_params() == {"n_neighbors": 7}
    new_model = clone(model)
    assert new_model.n_neighbors == 7 and not hasattr(new_model, "X_train_")


def test_predict_before_fit_raises_error():
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict(np.zeros((1, 2)))
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict_proba(np.zeros((1, 2)))


def test_x_and_y_with_different_sizes():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        KNeighborsClassifier().fit(np.zeros((4, 2)), np.zeros(3))


@pytest.mark.parametrize("k", [0, -1, 2.5, "3", None, True])
def test_invalid_n_neighbors(k):
    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="n_neighbors"):
        KNeighborsClassifier(n_neighbors=k).fit(X, y)


def test_n_neighbors_greater_than_n_samples():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="maior que o número de amostras"):
        KNeighborsClassifier(n_neighbors=4).fit(X, y)


def test_different_number_of_features_in_predict(blobs):
    X, y = blobs
    model = KNeighborsClassifier().fit(X, y)
    with pytest.raises(ValueError, match="mesmo número de features"):
        model.predict(np.zeros((2, 5)))


def test_x_with_nan_raises_error(blobs):
    X, y = blobs
    model = KNeighborsClassifier().fit(X, y)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.predict(X_bad)


# ---------------- Regressor ----------------


def test_regressor_mean_of_neighbors():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])
    model = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    # 0.4 -> vizinhos 0.0 e 1.0 -> média de 0 e 10
    np.testing.assert_allclose(model.predict(np.array([[0.4]])), [5.0])


def test_regressor_k1_reproduces_y_on_training_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 2))
    y = rng.normal(size=25)
    model = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y)


def test_regressor_k_equal_to_n_predicts_mean_of_y():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = KNeighborsRegressor(n_neighbors=6).fit(X, y)
    np.testing.assert_allclose(model.predict(np.array([[100.0]])), [3.5])


def test_regressor_perfect_r2_score_with_k1():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = X.ravel() ** 2
    model = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0)


def test_regressor_text_y_raises_error():
    with pytest.raises(ValueError, match="numérico"):
        KNeighborsRegressor(n_neighbors=1).fit(
            np.zeros((2, 1)), np.array(["a", "b"])
        )


def test_regressor_before_fit():
    with pytest.raises(NotFittedError):
        KNeighborsRegressor().predict(np.zeros((1, 1)))
