import numpy as np
import pytest

from statstech.discriminant_analysis.qda import QuadraticDiscriminantAnalysis
from statstech.utils.validation import NotFittedError


def test_fit():
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.2],
        [5.0, 6.0],
        [5.5, 5.8],
        [6.0, 6.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    result = model.fit(X, y)

    assert result is model
    assert np.array_equal(model.classes_, [0, 1])
    assert model.priors_.shape == (2,)
    assert model.means_.shape == (2, 2)
    assert model.covariances_.shape == (2, 2, 2)


def test_fit_reg_param_invalido():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=-1)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_reg_param_nan():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=np.nan)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_reg_param_infinito():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=np.inf)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_reg_param_booleano():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=True)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_uma_classe():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_classe_com_uma_observacao():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_quantidades_de_observacoes_diferentes():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)



def test_predict_proba():
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.2],
        [5.0, 6.0],
        [5.5, 5.8],
        [6.0, 6.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (6, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_proba_modelo_nao_treinado():
    X = np.array([[1.0, 2.0]])

    model = QuadraticDiscriminantAnalysis() 
    
    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_predict_proba_numero_incorreto_de_caracteristicas():
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.2],
        [5.0, 6.0],
        [5.5, 5.8],
        [6.0, 6.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    X_invalido = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        model.predict_proba(X_invalido)


def test_predict():
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.2],
        [5.0, 6.0],
        [5.5, 5.8],
        [6.0, 6.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (6,)
    assert np.all(np.isin(predictions, model.classes_))
    assert np.mean(predictions == y) >= 0.8


def test_predict_modelo_nao_treinado():
    X = np.array([[1.0, 2.0]])
    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_numero_incorreto_de_caracteristicas():
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.2],
        [5.0, 6.0],
        [5.5, 5.8],
        [6.0, 6.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    X_invalido = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        model.predict(X_invalido)
