"""Testes unitários para o classificador LinearDiscriminantAnalysis."""

import numpy as np
import pytest

from statstech.discriminant_analysis.lda import (
    LinearDiscriminantAnalysis,
)
from statstech.utils.validation import NotFittedError


def test_fit_multiclasse():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    assert np.array_equal(model.classes_, np.array([0, 1, 2]))
    assert model.priors_.shape == (3,)
    assert model.means_.shape == (3, 2)
    assert model.covariance_.shape == (2, 2)


def test_predict_proba_interval():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_proba_sum_one():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    for i in range(len(probabilities)):
        total = 0

        for j in range(len(model.classes_)):
            total += probabilities[i, j]

        assert abs(total - 1) < 1e-10


def test_predict_proba_format():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), len(model.classes_))


def test_predict_format():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)


def test_predict_choose_highest_probability():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    for i in range(len(X)):
        index = 0

        for j in range(1, len(model.classes_)):
            if probabilities[i, j] > probabilities[i, index]:
                index = j

        assert predictions[i] == model.classes_[index]


def test_predict_high_accuracy_separated_data():
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [9.0, 9.0],
        [9.0, 10.0],
        [10.0, 9.0],
    ])

    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    predictions = model.predict(X)

    correct = 0

    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct += 1

    accuracy = correct / len(y)

    assert accuracy >= 0.9


def test_predict_proba_model_not_fitted():
    model = LinearDiscriminantAnalysis()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(
        NotFittedError,
        match="ainda não foi treinado",
    ):
        model.predict_proba(X)


def test_fit_single_class():
    model = LinearDiscriminantAnalysis()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1, 1, 1])

    with pytest.raises(
        ValueError,
        match="pelo menos duas classes",
    ):
        model.fit(X, y)


def test_fit_samples_not_greater_than_classes():
    model = LinearDiscriminantAnalysis()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="número de amostras deve ser maior",
    ):
        model.fit(X, y)


def test_predict_proba_X_not_2d():
    model = LinearDiscriminantAnalysis()
    model.fit(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([0, 1, 1]),
    )

    X = np.array([1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="Esse array é 1D, precisa ser 2D",
    ):
        model.predict_proba(X)


def test_predict_proba_different_features():
    model = LinearDiscriminantAnalysis()
    model.fit(
        np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]),
        np.array([0, 1, 1]),
    )

    X = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(
        ValueError,
        match="mesmo número de características",
    ):
        model.predict_proba(X)


def test_priors_wrong_number_of_classes():
    model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])

    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError, match="um valor para cada classe"):
        model.fit(X, y)


def test_priors_not_sum_one():
    model = LinearDiscriminantAnalysis(priors=[0.2, 0.2, 0.2])

    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError, match="soma igual a 1"):
        model.fit(X, y)


def test_priors_non_positive():
    model = LinearDiscriminantAnalysis(priors=[0.5, 0.5, 0.0])

    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError, match="valores positivos e finitos"):
        model.fit(X, y)


def test_reg_param_negative():
    model = LinearDiscriminantAnalysis(reg_param=-0.1)

    X = np.array([[1.0], [2.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="não negativo"):
        model.fit(X, y)


def test_reg_param_invalid_type():
    model = LinearDiscriminantAnalysis(reg_param="abc")

    X = np.array([[1.0], [2.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="não negativo"):
        model.fit(X, y)


def test_reg_param_infinite():
    model = LinearDiscriminantAnalysis(reg_param=np.inf)

    X = np.array([[1.0], [2.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="não negativo"):
        model.fit(X, y)


def test_fit_covariance():
    X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1])

    model = LinearDiscriminantAnalysis(reg_param=0)
    model.fit(X, y)
    expected_covariance = np.array([
        [2.0, 2.0],
        [2.0, 2.0],
    ])

    assert np.allclose(model.covariance_, expected_covariance)


def test_fit_priors_unequal_classes():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
    ])

    y = np.array([0, 0, 0, 0, 1, 1])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    expected_priors = np.array([4 / 6, 2 / 6])

    assert np.allclose(model.priors_, expected_priors)
    assert np.isclose(model.priors_.sum(), 1.0)


def test_predict_high_accuracy_same_covariance_normals():
    rng = np.random.default_rng(42)

    mean_0 = np.array([0.0, 0.0])
    mean_1 = np.array([5.0, 5.0])

    covariance = np.array([
        [1.0, 0.3],
        [0.3, 1.0],
    ])

    X_0 = rng.multivariate_normal(mean_0, covariance, size=200)
    X_1 = rng.multivariate_normal(mean_1, covariance, size=200)

    X = np.vstack([X_0, X_1])
    y = np.array([0] * 200 + [1] * 200)

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)

    assert accuracy >= 0.95


def test_fit_means():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    y = np.array([0, 0, 1, 1])

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    expected_means = np.array([
        [2.0, 3.0],
        [6.0, 7.0],
    ])

    assert np.allclose(model.means_, expected_means)
