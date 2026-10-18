"""Testes unitários para o classificador LinearDiscriminantAnalysis."""

import numpy as np

from statstech.discriminant_analysis.lda import (
    LinearDiscriminantAnalysis,
)

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
