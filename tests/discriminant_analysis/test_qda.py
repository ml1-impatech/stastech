import numpy as np
import pytest

from statstech.discriminant_analysis.lda import LinearDiscriminantAnalysis
from statstech.discriminant_analysis.qda import QuadraticDiscriminantAnalysis
from statstech.utils.validation import NotFittedError


def test_fit():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [6.0, 6.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    result = model.fit(X, y)

    assert result is model
    assert np.array_equal(model.classes_, [0, 1])
    assert model.priors_.shape == (2,)
    assert model.means_.shape == (2, 2)
    assert model.covariances_.shape == (2, 2, 2)


def test_fit_invalid_reg_param():
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


def test_fit_infinite_reg_param():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=np.inf)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_boolean_reg_param():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=True)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_single_class():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_class_with_one_sample():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_mismatched_sample_counts():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1])

    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_proba():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [6.0, 6.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (6, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_proba_unfitted_model():
    X = np.array([[1.0, 2.0]])

    model = QuadraticDiscriminantAnalysis()
    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_predict_proba_wrong_number_of_features():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [6.0, 6.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    X_invalid = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        model.predict_proba(X_invalid)


def test_predict():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [6.0, 6.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (6,)
    assert np.all(np.isin(predictions, model.classes_))
    assert np.mean(predictions == y) >= 0.8


def test_predict_unfitted_model():
    X = np.array([[1.0, 2.0]])
    model = QuadraticDiscriminantAnalysis()

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_wrong_number_of_features():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [6.0, 6.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    X_invalid = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        model.predict(X_invalid)


def test_qda_with_different_covariances():
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

    split_index = int(0.7 * len(y))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()

    qda.fit(X_train, y_train)
    lda.fit(X_train, y_train)

    qda_accuracy = np.mean(qda.predict(X_test) == y_test)
    lda_accuracy = np.mean(lda.predict(X_test) == y_test)

    assert qda_accuracy > lda_accuracy


def test_qda_with_equal_covariances():
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


def test_predict_proba_with_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [5.0, 6.0],
            [6.0, 8.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_proba_with_singular_covariance():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [5.0, 6.0],
            [6.0, 8.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(reg_param=0.0)
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_fit_priors_and_means():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 1])

    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    assert np.allclose(model.priors_, [2 / 5, 3 / 5])

    mean_0 = np.mean(X[y == 0], axis=0)
    mean_1 = np.mean(X[y == 1], axis=0)

    assert np.allclose(model.means_[0], mean_0)
    assert np.allclose(model.means_[1], mean_1)


def test_fit_covariance_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [5.0, 6.0],
            [6.0, 8.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    reg_param = 0.1
    model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    model.fit(X, y)

    for k, class_label in enumerate(model.classes_):
        X_k = X[y == class_label]
        expected_covariance = np.cov(X_k, rowvar=False)
        expected_covariance += reg_param * np.eye(X.shape[1])

        assert np.allclose(
            model.covariances_[k],
            expected_covariance,
        )


@pytest.mark.parametrize(
    ("priors", "expected"),
    [
        (None, np.array([0.5, 0.5])),
        ([0.7, 0.3], np.array([0.7, 0.3])),
    ],
)
def test_priors(priors, expected):
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 5.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(priors=priors)
    model.fit(X, y)

    np.testing.assert_allclose(model.priors_, expected)


@pytest.mark.parametrize(
    "priors",
    [
        [0.5],
        [0.4, 0.3, 0.3],
        [-0.1, 1.1],
        [0.0, 1.0],
        [np.nan, 0.5],
        [np.inf, 0.0],
        [0.7, 0.4],
        ["a", 1.0],
    ],
)
def test_priors_invalid(priors):
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 5.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = QuadraticDiscriminantAnalysis(priors=priors)

    with pytest.raises(ValueError, match="priors"):
        model.fit(X, y)
