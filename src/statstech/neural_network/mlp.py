"""Rede neural MLP (perceptron multicamadas) para regressão."""

import warnings
from itertools import pairwise
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from statstech.base import BaseEstimator, RegressorMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)


def relu(preactivation: np.ndarray) -> np.ndarray:
    """Função de ativação ReLU.

    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.

    Returns
    -------
    activation : np.ndarray
        Array com o resultado da ativação.
    """
    activation = np.maximum(0, preactivation)
    return activation


def relu_derivative(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a derivada da função de ativação ReLU.

    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.

    Returns
    -------
    np.ndarray
        Array com a derivada (1.0 para x > 0, 0.0 caso contrário).
    """
    return (preactivation > 0).astype(float)


def identity(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a função de ativação identidade.

    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.

    Returns
    -------
    np.ndarray
        Cópia do array de entrada mantendo os valores originais.
    """
    return np.copy(preactivation)


def identity_derivative(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a derivada da função de ativação identidade.

    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.

    Returns
    -------
    np.ndarray
        Array de uns com a mesma estrutura da entrada.
    """
    return np.ones_like(preactivation, dtype=float)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "identity": (identity, identity_derivative),
}


class MLPRegressor(BaseEstimator, RegressorMixin):
    """Rede neural MLP para regressão.

    Parameters
    ----------
    hidden_layer_sizes : tuple de int, default=(32,)
        O número de neurônios em cada camada oculta.
    activation : {'relu', 'identity'}, default='relu'
        Função de ativação utilizada nas camadas ocultas.
    learning_rate : float, default=1e-3
        Taxa de aprendizado utilizada para a atualização dos pesos via
        gradiente descendente.
    max_iter : int, default=200
        Número de épocas de treinamento.
    batch_size : int, default=32
        Tamanho dos mini-batches para o treinamento.
    random_state : int, Generator ou None, default=None
        Semente para o gerador de números aleatórios.
    scale_y : bool, default=False
        Se True, o fit padroniza y (média 0, desvio 1) antes de treinar,
        e o predict devolve as previsões na escala original. Ajuda quando
        y está numa escala grande (por exemplo, notas em centenas), que
        faz o gradiente explodir.

    Attributes
    ----------
    weights_ : list de ndarray
        Matrizes de pesos, uma por camada.
    biases_ : list de ndarray
        Vetores de viés, um por camada.
    n_features_in_ : int
        Número de features vistas no fit.
    loss_curve_ : list de float
        MSE no conjunto de treino ao fim de cada época.
        Se scale_y=True, a perda fica na escala padronizada.
    y_mean_ : float
        Se scale_y=True, é a média de y,
        caso contrário é igual a 0.0.
    y_scale_ : float
        Se scale_y=True, é o desvio padrão de y,
        caso contrário é igual a 1.0.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        activation: str = "relu",
        learning_rate: float = 1e-3,
        max_iter: int = 200,
        batch_size: int = 32,
        random_state: int | np.random.Generator | None = None,
        scale_y: bool = False,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.scale_y = scale_y

    def _check_params(self) -> None:
        """Valida os hiperparâmetros.

        Raises
        ------
        ValueError
            Se algum hiperparâmetro tiver um valor inválido.
        """
        if not isinstance(self.scale_y, (bool, np.bool_)):
            raise ValueError("scale_y deve ser booleano.")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Ativação inválida {self.activation!r}. "
                f"As opções válidas são: {list(ACTIVATIONS)}"
            )
        if isinstance(self.learning_rate, bool) or not self.learning_rate > 0:
            raise ValueError("learning_rate deve ser positivo.")
        for name in ("max_iter", "batch_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} deve ser um inteiro positivo.")
        for n in self.hidden_layer_sizes:
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
                raise ValueError(
                    "hidden_layer_sizes deve conter só inteiros positivos."
                )

    def _init_weights(self, n_features: int, rng: np.random.Generator) -> None:
        """Inicializa pesos e vieses de todas as camadas."""
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        weights = []
        biases = []

        for n_in, n_out in pairwise(layer_sizes):
            scale = np.sqrt(2 / n_in)
            weights.append(rng.normal(0, scale, size=(n_in, n_out)))
            biases.append(np.zeros(n_out))

        self.weights_ = weights
        self.biases_ = biases

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Treina a rede com gradiente descendente em mini-lotes.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : array-like de formato (n_amostras,)
            Valores alvo.

        Returns
        -------
        self : MLPRegressor
            O modelo treinado.

        Raises
        ------
        ValueError
            Se algum hiperparâmetro for inválido, ou
            se X ou y forem inválidos.
        """
        self._check_params()
        X, y = check_X_y(X, y)
        X = X.astype(float)
        y = y.astype(float)

        if self.scale_y:
            self.y_mean_ = float(np.mean(y))
            std = float(np.std(y))
            self.y_scale_ = std if std > 0 else 1.0
        else:
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0
        y = (y - self.y_mean_) / self.y_scale_

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self._init_weights(n_features, rng)
        self.loss_curve_ = []

        last_weights = [W.copy() for W in self.weights_]
        last_biases = [b.copy() for b in self.biases_]

        for _ in range(self.max_iter):
            order = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                zs, activations = self._forward(X_batch)
                grad_W, grad_b = self._backward(y_batch, zs, activations)

                for layer in range(len(self.weights_)):
                    self.weights_[layer] -= self.learning_rate * grad_W[layer]
                    self.biases_[layer] -= self.learning_rate * grad_b[layer]

            _, activations = self._forward(X)
            loss = np.mean((activations[-1][:, 0] - y) ** 2)
            if not np.isfinite(loss):
                self.weights_ = last_weights
                self.biases_ = last_biases
                warnings.warn(
                    "A loss virou NaN ou infinito; treino interrompido e "
                    "pesos da última época válida restaurados. "
                    "Diminua learning_rate, padronize X ou use scale_y=True.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            self.loss_curve_.append(loss)
            last_weights = [W.copy() for W in self.weights_]
            last_biases = [b.copy() for b in self.biases_]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Prevê y para cada amostra de X.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados para os quais a previsão será feita.

        Returns
        -------
        y_pred : ndarray de formato (n_amostras,)
            Valores previstos, um por amostra.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X não for 2D, tiver NaN ou infinitos, ou tiver
            número de features diferente do usado no treino.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        check_n_features(X, self.weights_[0].shape[0])
        _, activations = self._forward(X)
        y_pred = activations[-1][:, 0]
        return y_pred * self.y_scale_ + self.y_mean_

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Propaga X pela rede, camada por camada."""
        func, _ = ACTIVATIONS[self.activation]
        a = X
        zs = []
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b  # a: (n, h_in), W: (h_in, h_out) -> z: (n, h_out)
            if i == len(self.weights_) - 1:
                a = identity(z)
            else:
                a = func(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def _backward(
        self,
        y: np.ndarray,
        zs: list[np.ndarray],
        activations: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Calcula os gradientes da loss por backpropagation."""
        n = len(y)
        _, der = ACTIVATIONS[self.activation]
        y_pred = activations[-1]
        y = y.reshape(-1, 1)
        delta = (2 / n) * (y_pred - y) * identity_derivative(zs[-1])
        grad_W = [np.zeros_like(W) for W in self.weights_]
        grad_b = [np.zeros_like(b) for b in self.biases_]
        for layer in range(len(self.weights_) - 1, -1, -1):
            grad_W[layer] = activations[layer].T @ delta  # (h_in, h_out)
            grad_b[layer] = delta.sum(axis=0)  # (h_out,)
            if layer != 0:
                delta = delta @ self.weights_[layer].T * der(zs[layer - 1])
        return grad_W, grad_b
