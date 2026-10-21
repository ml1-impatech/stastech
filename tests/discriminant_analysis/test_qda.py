import numpy as np
import pytest

from statstech.discriminant_analysis.qda import QuadraticDiscriminantAnalysis
from statstech.utils.validation import NotFittedError
from statstech.discriminant_analysis.lda import LinearDiscriminantAnalysis


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


def test_qda_com_covariancias_diferentes():
    rng = np.random.default_rng(42)

    X0 = rng.multivariate_normal(
        mean=[0, 0],
        cov=[[3, 0], [0, 0.3]],
        size=400,
    )
    X1 = rng.multivariate_normal(
        mean=[0, 0],
        cov=[[0.3, 0], [0, 3]],
        size=400,
    )

    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    indices = rng.permutation(len(y))
    X, y = X[indices], y[indices]

    corte = int(0.7 * len(y))
    X_treino, X_teste = X[:corte], X[corte:]
    y_treino, y_teste = y[:corte], y[corte:]

    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()

    qda.fit(X_treino, y_treino)
    lda.fit(X_treino, y_treino)

    acuracia_qda = np.mean(qda.predict(X_teste) == y_teste)
    acuracia_lda = np.mean(lda.predict(X_teste) == y_teste)

    assert acuracia_qda > acuracia_lda


def test_qda_com_covariancias_iguais():
    rng = np.random.default_rng(42)

    cov = [[1.0, 0.3], [0.3, 1.5]]

    X0 = rng.multivariate_normal([-2, 0], cov, size=500)
    X1 = rng.multivariate_normal([2, 0], cov, size=500)

    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()

    qda.fit(X, y)
    lda.fit(X, y)

    pred_qda = qda.predict(X)
    pred_lda = lda.predict(X)

    assert np.mean(pred_qda == pred_lda) >= 0.95


def test_predict_proba_com_regularizacao():
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [5.0, 6.0],
        [6.0, 8.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)