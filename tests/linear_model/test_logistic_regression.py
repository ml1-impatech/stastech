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

def test_gradient_finite_difference():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    X_scaled = (X - model.mean_) / model.std_
    Y = np.zeros((len(y), len(model.classes_)))

    for i in range(len(y)):
        Y[i, model.classes_ == y[i]] = 1

    coef = np.zeros(model.coef_.shape)
    intercept = np.zeros(model.intercept_.shape)

    _, gradient, _ = model._loss_and_gradient(X_scaled, Y, coef, intercept)

    # pertubação p/ o cálculo numérico
    h = 1e-5

    numerical_gradient = np.zeros(coef.shape)

    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            coef_incr = coef.copy()
            coef_decr = coef.copy()

            coef_incr[i, j] += h 
            coef_decr[i, j] -= h 

            loss_incr, _, _ = model._loss_and_gradient(X_scaled, Y, coef_incr, intercept)
            loss_decr, _, _ = model._loss_and_gradient(X_scaled, Y, coef_decr, intercept)

            numerical_gradient[i, j] = ( loss_incr - loss_decr) / (2 * h)

    differ = np.abs(gradient - numerical_gradient)

    for i in range(differ.shape[0]):
        for j in range(differ.shape[1]):
            assert differ[i, j] < 1e-5

def test_predict_proba_interval():
    X = np.array([
             [1.0, 2.0],
             [2.0, 1.0],
             [4.0, 5.0],
             [5.0, 4.0],
             [8.0, 9.0],
             [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    for i in range(len(prob)):
        for j in range(len(model.classes_)):
            assert prob[i, j] >= 0
            assert prob[i, j] <= 1

def test_predict_proba_sum_one():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    for i in range(len(prob)):
        total = 0 

        for j in range(len(model.classes_)):
            total += prob[i, j]

        assert abs(total - 1) < 1e-10


def test_predict_proba_format():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    assert prob.shape == (len(X), len(model.classes_))

def test_predict_text_labels():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array(["não", "não", "sim", "sim", "talvez", "talvez"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    for i in range(len(predictions)):
        assert predictions[i] in model.classes_

def test_predict_format():
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

    predictions = model.predict(X)

    assert predictions.shape == (len(X), )






            



    


