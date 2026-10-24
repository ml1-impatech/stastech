"""Implementação do classificador de regressão logística."""

import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin
from statstech.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Classificador de regressão logística multiclasse com regularização L2.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Taxa de aprendizado usada na descida do gradiente.
    max_iter : int, default=1000
        Número máximo de iterações do treinamento.
    tol : float, default=1e-6
        Tolerância para o critério de convergência.
    fit_intercept : bool, default=True
        Indica se um termo independente deve ser ajustado.
    alpha : float, default=0.01
        Coeficiente de regularização L2.

    Attributes
    ----------
    classes_ : ndarray
        Classes encontradas durante o treinamento.
    coef_ : ndarray
        Matriz de coeficientes estimados.
    intercept_ : ndarray
        Vetor de interceptos estimados, ou vetor de zeros quando
        fit_intercept=False.
    loss_history_ : list
        Valores da função de perda registrados durante o treinamento.

    Notes
    -----
    O pré-processamento das características, como a padronização, não é
    realizado automaticamente. Logo, é recomendado avaliar a escala
    das variáveis antes do treinamento.
    """

    def __init__(
        self,
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-6,
        fit_intercept=True,
        alpha=0.01,
    ):
        """Guarda os hiperparâmetros."""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def _check_params(self) -> None:
        """Valida os hiperparâmetros do classificador.

        Raises
        ------
        ValueError
            Se algum parâmetro tiver um valor inválido.
        """
        if (
            not isinstance(self.learning_rate, (int, float, np.integer, np.floating))
            or isinstance(self.learning_rate, (bool, np.bool_))
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate deve ser positivo e finito.")

        if (
            not isinstance(self.max_iter, (int, np.integer))
            or isinstance(self.max_iter, (bool, np.bool_))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter deve ser um inteiro positivo.")

        if (
            not isinstance(self.tol, (int, float, np.integer, np.floating))
            or isinstance(self.tol, (bool, np.bool_))
            or not np.isfinite(self.tol)
            or self.tol < 0
        ):
            raise ValueError("tol deve ser não negativo e finito.")

        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise TypeError("fit_intercept deve ser booleano.")

        if (
            not isinstance(self.alpha, (int, float, np.integer, np.floating))
            or isinstance(self.alpha, (bool, np.bool_))
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError("alpha deve ser não negativo e finito.")

    def _loss_and_gradient(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray | None,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Calcula a função de perda e seus gradientes.

        Parameters
        ----------
        X : ndarray
            Matriz de características, com dimensão
            (n_amostras, n_features).
        Y : ndarray
            Matriz indicadora das classes, com dimensão
            (n_amostras, n_classes).
        coef : ndarray
            Matriz atual de coeficientes.
        intercept : ndarray or None
            Vetor atual de interceptos, ou None se não for utilizado.

        Returns
        -------
        loss : float
            Valor da função de perda com regularização L2.
        gradient : ndarray
            Gradiente da perda em relação aos coeficientes.
        intercept_gradient : ndarray or None
            Gradiente em relação aos interceptos, ou None se não utilizados.

        Raises
        ------
        ValueError
            Se X e Y tiverem números diferentes de observações.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X e Y devem ter o mesmo número de observações.")

        n = X.shape[0]

        if self.fit_intercept:
            z = X @ coef + intercept
        else:
            z = X @ coef

        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)
        P = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        P = np.clip(P, 1e-15, 1.0)

        loss = -np.sum(Y * np.log(P)) / n

        if self.alpha > 0:
            L2 = (self.alpha / (2 * n)) * np.sum(coef**2)
            loss = loss + L2

        gradient = (X.T @ (P - Y)) / n

        if self.alpha > 0:
            gradient = gradient + (self.alpha / n) * coef

        if self.fit_intercept:
            intercept_gradient = np.mean(P - Y, axis=0)
        else:
            intercept_gradient = None

        return loss, gradient, intercept_gradient

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "LogisticRegression":
        """Ajusta o modelo aos dados de treinamento.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Matriz de características, com uma linha por amostra
            e uma coluna por característica.
        y : ndarray
            Vetor unidimensional com os rótulos das classes.

        Returns
        -------
        self : LogisticRegression
            Instância ajustada do classificador.

        Raises
        ------
        ValueError
            Se os dados ou algum hiperparâmetro tiverem valores inválidos.
            Se as dimensões dos dados forem inválidas, X estiver vazio,
            X contiver valores não finitos, y contiver NaN ou houver
            menos de duas classes.
        """
        self._check_params()
        X, y = check_X_y(X, y)

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("y deve ter pelo menos duas classes.")

        K = len(classes)
        p = X.shape[1]
        n = X.shape[0]

        y_indices = np.searchsorted(classes, y)

        Y = np.zeros((n, K))
        Y[np.arange(n), y_indices] = 1

        coef = np.zeros((p, K))
        intercept = np.zeros(K) if self.fit_intercept else None
        loss_history: list[float] = []

        for iteration in range(self.max_iter):
            loss, gradient, intercept_gradient = self._loss_and_gradient(
                X, Y, coef, intercept
            )

            # critério de convergência
            if len(loss_history) > 0:
                variation = abs(loss - loss_history[-1])
                loss_history.append(loss)

                if variation < self.tol:
                    break
            else:
                loss_history.append(loss)

            # atualiza o intercepto
            if self.fit_intercept:
                intercept -= self.learning_rate * intercept_gradient

            # atualiza os coeficientes
            coef -= self.learning_rate * gradient

        self.classes_ = classes
        self.coef_ = coef

        if self.fit_intercept:
            self.intercept_ = intercept
        else:
            self.intercept_ = np.zeros(K)

        self.loss_history_ = loss_history

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula as probabilidades previstas para cada classe.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Matriz de características, com uma linha por amostra
            e uma coluna por característica.

        Returns
        -------
        prob : ndarray
            Matriz de probabilidades, com uma coluna por classe.

        Raises
        ------
        NotFittedError
            Se o modelo não tiver sido treinado.
        ValueError
            Se X tiver dimensões incompatíveis, estiver vazio
            ou contiver valores não finitos.
        """
        check_is_fitted(self, attributes=["coef_", "classes_"])
        X = check_array(X)

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "X deve ter o mesmo número de características usado no fit."
            )
        
        if self.intercept_ is None:
            raise RuntimeError("O intercepto não foi inicializado.")

        z = X @ self.coef_ + self.intercept_
        # evita que a exponencial exploda
        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)
        prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prevê a classe de cada observação.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Matriz bidimensional de características, com uma linha por amostra
            e uma coluna por característica.

        Returns
        -------
        ndarray
            Vetor com a classe prevista para cada observação.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        ValueError
            Se X tiver dimensões incompatíveis, estiver vazio
            ou contiver valores não finitos.
        """
        prob = self.predict_proba(X)
        predictions = np.argmax(prob, axis=1)

        return self.classes_[predictions]
