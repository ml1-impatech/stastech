import numpy as np 

from statstech.linear_model.logistic_regression import LogisticRegression

def test_fit_multiclasse():
    X = np.array([
        [1.0, 2.0], 
        [2.0, 1.0],
        [4.0, 5.0], 
        [5.0, 4.0], 
        [8.0, 9.0], 
        [9.0, 8.0] 

    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter = 100)
    model.fit(X, y)

    assert np.array_equal(model.classes_, np.array([0, 1, 2]))
    assert model.coef_.shape == (2, 3)
    assert model.intercept_.shape == (3,)
    assert len(model.loss_history_) > 0

def test_loss_decreases():
    X = np.array([
        [1.0, 2.0], 
        [2.0, 1.0],
        [4.0, 5.0], 
        [5.0, 4.0], 
        [8.0, 9.0], 
        [9.0, 8.0] 

    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter = 100)
    model.fit(X, y)

    assert model.loss_history_[-1] < model.loss_history_[0]

def test_fit_respect_max_iter():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter = 5, tol = 0)
    model.fit(X, y)

    assert len(model.loss_history_) == 5

def test_fit_convergence_criterion():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100, tol=100)
    model.fit(X, y)

    assert len(model.loss_history_) == 2


