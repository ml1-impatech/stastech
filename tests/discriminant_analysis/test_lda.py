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

