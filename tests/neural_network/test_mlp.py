import numpy as np
import pytest

from statstech.base import clone
from statstech.metrics.regression import r2_score
from statstech.neural_network.mlp import (
    MLPRegressor,
    identity,
    identity_derivative,
    relu,
    relu_derivative,
)
from statstech.utils.validation import NotFittedError


def test_get_params_and_clone():
    model = MLPRegressor(hidden_layer_sizes=(8, 4), learning_rate=0.01)
    new = clone(model)
    assert new is not model
    assert new.get_params() == model.get_params()


def test_relu():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 3.0])
    np.testing.assert_allclose(relu(z), expected)


def test_relu_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(relu_derivative(z), expected)


def test_identity():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(identity(z), expected)


def test_identity_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(identity_derivative(z), expected)


def test_activations_keep_shape():
    z = np.array([[-1.0, 2.0], [3.0, -4.0]])
    for func in [relu, relu_derivative, identity, identity_derivative]:
        assert func(z).shape == z.shape


def test_init_weights_shapes():
    model = MLPRegressor(hidden_layer_sizes=(32, 16))
    model._init_weights(5, np.random.default_rng(0))
    assert [w.shape for w in model.weights_] == [(5, 32), (32, 16), (16, 1)]
    assert [b.shape for b in model.biases_] == [(32,), (16,), (1,)]


def test_init_weights_biases():
    model = MLPRegressor(hidden_layer_sizes=(32, 16))
    model._init_weights(5, np.random.default_rng(0))
    for b in model.biases_:
        np.testing.assert_allclose(b, 0)


def test_forward_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([-10.0])]
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    zs, activations = model._forward(X)
    np.testing.assert_allclose(activations[-1], [[-5.0], [-6.0]])


def test_predict_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([1.0])]
    model.y_mean_ = 0.0
    model.y_scale_ = 1.0
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, [6.0, 5.0])
    assert y_pred.shape == (2,)


def test_predict_with_hidden_layer():
    model = MLPRegressor(hidden_layer_sizes=(3,))
    model.weights_ = [
        np.array([[1.0, -1.0, 0.0], [2.0, 1.0, 1.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ]
    model.biases_ = [np.array([0.0, -3.0, 0.0]), np.array([0.5])]
    model.y_mean_ = 0.0
    model.y_scale_ = 1.0
    X = np.array([[1.0, 2.0]])
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, [7.5])


def test_backward_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([1.0])]
    X = np.array([[1.0, 1.0]])
    y = np.array([4.0])
    zs, activations = model._forward(X)
    grad_W, grad_b = model._backward(y, zs, activations)
    np.testing.assert_allclose(grad_W[0], [[4.0], [4.0]])
    np.testing.assert_allclose(grad_b[0], [4.0])


def test_backward_with_hidden_layer():
    model = MLPRegressor(hidden_layer_sizes=(3,))
    model.weights_ = [
        np.array([[1.0, -1.0, 0.0], [2.0, 1.0, 1.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ]
    model.biases_ = [np.array([0.0, -3.0, 0.0]), np.array([0.5])]
    model.y_mean_ = 0.0
    model.y_scale_ = 1.0
    X = np.array([[1.0, 2.0]])
    y = np.array([5.5])
    zs, activations = model._forward(X)
    grad_W, grad_b = model._backward(y, zs, activations)
    np.testing.assert_allclose(grad_W[0], [[4.0, 0.0, 4.0], [8.0, 0.0, 8.0]])
    np.testing.assert_allclose(grad_b[0], [4.0, 0.0, 4.0])
    np.testing.assert_allclose(grad_W[1], [[20.0], [0.0], [8.0]])
    np.testing.assert_allclose(grad_b[1], [4.0])


def test_fit_returns_self():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1, random_state=0)
    assert model.fit(X, y) is model


def test_fit_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        max_iter=200,
        learning_rate=1e-2,
        random_state=0,
    )
    model.fit(X, y)
    assert model.loss_curve_[-1] < model.loss_curve_[0]


def test_fit_same_seed_same_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model_a = MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=0)
    model_b = MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=0)
    model_a.fit(X, y)
    model_b.fit(X, y)
    for wa, wb in zip(model_a.weights_, model_b.weights_):
        np.testing.assert_allclose(wa, wb)
    for ba, bb in zip(model_a.biases_, model_b.biases_):
        np.testing.assert_allclose(ba, bb)


def test_predict_before_fit_raises():
    model = MLPRegressor()
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 2)))


def test_predict_wrong_n_features_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1, random_state=0)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 3)))


def make_large_target_data(n=300):
    """y em torno de 500, com desvio perto de 100, como as notas do ENEM."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = 500 + 100 * X[:, 0] + 5 * rng.normal(size=n)
    return X, y


def test_scale_y_false_stores_identity_scaling():
    X, y = make_large_target_data(n=40)
    model = MLPRegressor(max_iter=1, random_state=0).fit(X, y / 100)
    assert model.y_mean_ == 0.0
    assert model.y_scale_ == 1.0


def test_scale_y_true_stores_mean_and_std():
    X, y = make_large_target_data(n=40)
    model = MLPRegressor(max_iter=1, scale_y=True, random_state=0).fit(X, y)
    assert model.y_mean_ == pytest.approx(np.mean(y))
    assert model.y_scale_ == pytest.approx(np.std(y))


def test_scale_y_equals_training_on_standardized_y():
    X, y = make_large_target_data(n=60)
    mean, std = y.mean(), y.std()
    params = {"hidden_layer_sizes": (4,), "max_iter": 5, "random_state": 0}

    scaled = MLPRegressor(scale_y=True, **params).fit(X, y)
    manual = MLPRegressor(**params).fit(X, (y - mean) / std)

    np.testing.assert_allclose(scaled.predict(X), manual.predict(X) * std + mean)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scale_y_learns_target_in_hundreds():
    X, y = make_large_target_data()
    params = {"max_iter": 100, "learning_rate": 1e-2, "random_state": 0}

    with_scaling = MLPRegressor(scale_y=True, **params).fit(X, y)
    without_scaling = MLPRegressor(**params).fit(X, y)

    assert r2_score(y, with_scaling.predict(X)) > 0.95
    assert with_scaling.predict(X).mean() == pytest.approx(y.mean(), abs=5)
    assert r2_score(y, without_scaling.predict(X)) < 0


def test_scale_y_loss_curve_is_in_standardized_scale():
    X, y = make_large_target_data()
    model = MLPRegressor(
        max_iter=50, learning_rate=1e-2, scale_y=True, random_state=0
    ).fit(X, y)
    assert model.loss_curve_[-1] < 1


def test_scale_y_constant_target_does_not_divide_by_zero():
    X, _ = make_large_target_data(n=40)
    y = np.full(40, 7.0)
    model = MLPRegressor(max_iter=5, scale_y=True, random_state=0).fit(X, y)
    assert model.y_mean_ == 7.0
    assert model.y_scale_ == 1.0
    assert np.isfinite(model.predict(X)).all()


def test_scale_y_accepts_numpy_bool():
    X, y = make_large_target_data(n=40)
    model = MLPRegressor(max_iter=1, scale_y=np.True_, random_state=0).fit(X, y)
    assert model.y_mean_ == pytest.approx(np.mean(y))


@pytest.mark.parametrize("scale_y", ["sim", 1, None])
def test_invalid_scale_y_raises(scale_y):
    X, y = make_large_target_data(n=40)
    with pytest.raises(ValueError, match="scale_y"):
        MLPRegressor(max_iter=1, scale_y=scale_y).fit(X, y)


def test_diverging_training_restores_last_valid_epoch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = 100 * X[:, 0]
    params = {"learning_rate": 0.02, "random_state": 0}

    with pytest.warns(RuntimeWarning, match="restaurados"):
        model = MLPRegressor(max_iter=10, **params).fit(X, y)
    n_valid = len(model.loss_curve_)
    assert 0 < n_valid < 10

    reference = MLPRegressor(max_iter=n_valid, **params).fit(X, y)
    for W, W_ref in zip(model.weights_, reference.weights_):
        np.testing.assert_allclose(W, W_ref)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_diverging_in_first_epoch_restores_initial_weights():
    X, y = make_large_target_data(n=100)
    model = MLPRegressor(learning_rate=10, max_iter=5, random_state=0).fit(X, y)
    assert model.loss_curve_ == []

    reference = MLPRegressor(random_state=0)
    reference._init_weights(X.shape[1], np.random.default_rng(0))
    for W, W_ref in zip(model.weights_, reference.weights_):
        np.testing.assert_allclose(W, W_ref)
    assert np.isfinite(model.predict(X)).all()