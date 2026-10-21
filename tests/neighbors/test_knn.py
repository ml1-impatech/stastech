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


def test_exemplo_feito_a_mao():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 0, 1, 1])
    modelo = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    # 1.4 -> vizinhos 1.0, 2.0, 0.0 (todos da classe 0)
    # 9.0 -> vizinhos 10.0, 11.0, 2.0 (dois votos para a classe 1)
    pred = modelo.predict(np.array([[1.4], [9.0]]))
    assert np.array_equal(pred, [0, 1])


def test_k1_acerta_100_por_cento_no_treino():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, size=30)
    modelo = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert modelo.score(X, y) == 1.0


def test_k_igual_ao_numero_de_amostras_preve_classe_majoritaria():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 0, 0])
    modelo = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    pred = modelo.predict(np.array([[100.0], [-50.0], [3.5]]))
    assert np.all(pred == 1)


def test_rotulos_de_texto_voltam_como_texto():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["nao", "nao", "sim", "sim"])
    modelo = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    pred = modelo.predict(np.array([[0.2], [10.4]]))
    assert list(pred) == ["nao", "sim"]
    assert list(modelo.classes_) == ["nao", "sim"]


def test_predict_proba_em_0_1_e_soma_1(blobs):
    X, y = blobs
    modelo = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    proba = modelo.predict_proba(X)
    assert proba.shape == (20, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_proporcao_de_votos():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    modelo = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    # vizinhos de 0.5: 0.0 (c0), 1.0 (c0), 2.0 (c1) -> [2/3, 1/3]
    proba = modelo.predict_proba(np.array([[0.5]]))
    np.testing.assert_allclose(proba, [[2 / 3, 1 / 3]])


def test_empate_no_voto_vence_a_menor_classe():
    X = np.array([[0.0], [2.0]])
    y = np.array([5, 3])
    modelo = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    # 1 voto para cada classe: vence a menor (3)
    assert modelo.predict(np.array([[1.0]]))[0] == 3


def test_multiclasse(blobs):
    X, y = blobs
    X3 = np.vstack([X, X[:10] + 20.0])
    y3 = np.concatenate([y, np.full(10, 2)])
    modelo = KNeighborsClassifier(n_neighbors=3).fit(X3, y3)
    assert modelo.score(X3, y3) == 1.0
    assert modelo.predict_proba(X3).shape == (30, 3)


def test_classes_ordenadas_e_guarda_treino(blobs):
    X, y = blobs
    modelo = KNeighborsClassifier().fit(X, y)
    assert np.array_equal(modelo.classes_, [0, 1])
    assert modelo.X_train_.shape == X.shape
    assert modelo.y_train_.shape == y.shape


def test_fit_devolve_self_e_copia_os_dados(blobs):
    X, y = blobs
    modelo = KNeighborsClassifier()
    assert modelo.fit(X, y) is modelo
    X_original = X.copy()
    X[0, 0] = 999.0  # mexer no X do usuário não pode afetar o modelo
    np.testing.assert_allclose(modelo.X_train_[0], X_original[0])


def test_get_params_e_clone():
    from statstech.base import clone

    modelo = KNeighborsClassifier(n_neighbors=7)
    assert modelo.get_params() == {"n_neighbors": 7}
    novo = clone(modelo)
    assert novo.n_neighbors == 7 and not hasattr(novo, "X_train_")


def test_predict_antes_do_fit_levanta_erro():
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict(np.zeros((1, 2)))
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict_proba(np.zeros((1, 2)))


def test_x_e_y_com_tamanhos_diferentes():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        KNeighborsClassifier().fit(np.zeros((4, 2)), np.zeros(3))


@pytest.mark.parametrize("k", [0, -1, 2.5, "3", None, True])
def test_n_neighbors_invalido(k):
    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="n_neighbors"):
        KNeighborsClassifier(n_neighbors=k).fit(X, y)


def test_n_neighbors_maior_que_amostras():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="maior que o número de amostras"):
        KNeighborsClassifier(n_neighbors=4).fit(X, y)


def test_numero_de_features_diferente_no_predict(blobs):
    X, y = blobs
    modelo = KNeighborsClassifier().fit(X, y)
    with pytest.raises(ValueError, match="mesmo número de features"):
        modelo.predict(np.zeros((2, 5)))


def test_x_com_nan_levanta_erro(blobs):
    X, y = blobs
    modelo = KNeighborsClassifier().fit(X, y)
    X_ruim = X.copy()
    X_ruim[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        modelo.predict(X_ruim)


# ---------------- Regressor ----------------


def test_regressor_media_dos_vizinhos():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])
    modelo = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    # 0.4 -> vizinhos 0.0 e 1.0 -> média de 0 e 10
    np.testing.assert_allclose(modelo.predict(np.array([[0.4]])), [5.0])


def test_regressor_k1_reproduz_y_no_treino():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 2))
    y = rng.normal(size=25)
    modelo = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    np.testing.assert_allclose(modelo.predict(X), y)


def test_regressor_k_igual_a_n_preve_a_media_de_y():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    modelo = KNeighborsRegressor(n_neighbors=6).fit(X, y)
    np.testing.assert_allclose(modelo.predict(np.array([[100.0]])), [3.5])


def test_regressor_score_r2_perfeito_com_k1():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = X.ravel() ** 2
    modelo = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    assert modelo.score(X, y) == pytest.approx(1.0)


def test_regressor_y_de_texto_levanta_erro():
    with pytest.raises(ValueError, match="numérico"):
        KNeighborsRegressor(n_neighbors=1).fit(
            np.zeros((2, 1)), np.array(["a", "b"])
        )


def test_regressor_antes_do_fit():
    with pytest.raises(NotFittedError):
        KNeighborsRegressor().predict(np.zeros((1, 1)))
