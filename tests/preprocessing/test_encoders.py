"""Testes de LabelEncoder e OneHotEncoder."""

import numpy as np
import pytest

from statstech.preprocessing.encoders import LabelEncoder, OneHotEncoder
from statstech.utils.validation import NotFittedError

# ---------------- LabelEncoder ----------------


def test_label_exemplo_feito_a_mao():
    enc = LabelEncoder().fit(["gato", "cão", "gato", "pássaro"])
    assert list(enc.classes_) == ["cão", "gato", "pássaro"]
    assert list(enc.transform(["gato", "cão", "pássaro"])) == [1, 0, 2]


def test_label_inverse_transform_volta_ao_original():
    y = np.array(["sim", "nao", "sim", "nao", "sim"])
    enc = LabelEncoder()
    codigos = enc.fit_transform(y)
    assert np.array_equal(enc.inverse_transform(codigos), y)


def test_label_com_numeros():
    enc = LabelEncoder()
    codigos = enc.fit_transform([10, 5, 10, 7])
    assert list(codigos) == [2, 0, 2, 1]
    assert list(enc.inverse_transform(codigos)) == [10, 5, 10, 7]


def test_label_rotulo_desconhecido_levanta_erro():
    enc = LabelEncoder().fit(["a", "b"])
    with pytest.raises(ValueError, match="não vistos"):
        enc.transform(["a", "c"])


def test_label_codigo_invalido_no_inverse():
    enc = LabelEncoder().fit(["a", "b"])
    with pytest.raises(ValueError, match="entre 0 e 1"):
        enc.inverse_transform([0, 2])
    with pytest.raises(ValueError, match="inteiros"):
        enc.inverse_transform([0.5])


def test_label_entradas_invalidas():
    enc = LabelEncoder()
    with pytest.raises(NotFittedError):
        enc.transform(["a"])
    with pytest.raises(ValueError, match="1D"):
        enc.fit([["a"], ["b"]])
    with pytest.raises(ValueError, match="vazio"):
        enc.fit([])


# ---------------- OneHotEncoder ----------------


@pytest.fixture
def X():
    return np.array([["azul", "P"], ["verde", "G"], ["azul", "M"],
                     ["vermelho", "P"]])


def test_onehot_exemplo_feito_a_mao():
    enc = OneHotEncoder().fit([["a"], ["b"], ["c"]])
    Z = enc.transform([["b"], ["a"], ["c"]])
    esperado = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_array_equal(Z, esperado)


def test_onehot_formato_e_categorias(X):
    enc = OneHotEncoder().fit(X)
    Z = enc.transform(X)
    assert Z.shape == (4, 6)  # 3 cores + 3 tamanhos
    assert enc.n_features_out_ == 6
    assert enc.n_features_in_ == 2
    assert list(enc.categories_[0]) == ["azul", "verde", "vermelho"]
    assert list(enc.categories_[1]) == ["G", "M", "P"]


def test_onehot_um_unico_1_por_variavel(X):
    Z = OneHotEncoder().fit_transform(X)
    np.testing.assert_array_equal(Z[:, :3].sum(axis=1), 1.0)  # cor
    np.testing.assert_array_equal(Z[:, 3:].sum(axis=1), 1.0)  # tamanho
    assert set(np.unique(Z)) == {0.0, 1.0}


def test_onehot_drop_first(X):
    Z = OneHotEncoder(drop="first").fit_transform(X)
    assert Z.shape == (4, 4)  # (3-1) + (3-1)
    # "azul" é a primeira cor: linha 0 fica com zeros na parte da cor
    np.testing.assert_array_equal(Z[0, :2], [0.0, 0.0])


def test_onehot_categoria_desconhecida_com_error(X):
    enc = OneHotEncoder().fit(X)
    with pytest.raises(ValueError, match="não vistas"):
        enc.transform([["roxo", "P"]])


def test_onehot_categoria_desconhecida_com_ignore(X):
    enc = OneHotEncoder(handle_unknown="ignore").fit(X)
    Z = enc.transform([["roxo", "P"]])
    np.testing.assert_array_equal(Z[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(Z[0, 3:], [0.0, 0.0, 1.0])


def test_onehot_com_numeros():
    Z = OneHotEncoder().fit_transform([[1], [2], [1]])
    np.testing.assert_array_equal(Z, [[1, 0], [0, 1], [1, 0]])


def test_onehot_parametros_e_entradas_invalidas(X):
    with pytest.raises(ValueError, match="handle_unknown"):
        OneHotEncoder(handle_unknown="x").fit(X)
    with pytest.raises(ValueError, match="drop"):
        OneHotEncoder(drop="last").fit(X)
    with pytest.raises(NotFittedError):
        OneHotEncoder().transform(X)
    enc = OneHotEncoder().fit(X)
    with pytest.raises(ValueError, match="features"):
        enc.transform([["azul"]])
    with pytest.raises(ValueError, match="2D"):
        OneHotEncoder().fit(["a", "b"])
