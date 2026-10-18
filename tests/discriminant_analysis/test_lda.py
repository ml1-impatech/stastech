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


