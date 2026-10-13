import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Implementa a regressão logística."""

    def __init__(
            self,
            learning_rate = 0.1,
            max_iter = 1000,
            tol = 1e-6, 
            fit_intercept = True,
            lambda_ = 0.01):

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo e finito.")

        if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter <= 0:
            raise ValueError("max_iter deve ser um inteiro positivo.")

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol deve ser não negativo e finito.")

        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept deve ser booleano.")

        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("lambda_ deve ser não negativo e finito.")
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.lambda_ = lambda_

    # função auxiliar para facilitar os testes do gradiente 
    def _loss_and_gradient(self, X, Y, coef, intercept):
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

        if self.lambda_ > 0:
            L2 = (self.lambda_ / (2*n)) * np.sum(coef ** 2)
            loss = loss + L2

        gradient = ( X.T @ (P - Y) ) / n 
    
        if self.lambda_ > 0:
            gradient = gradient + (self.lambda_ / n) * coef
        
        if self.fit_intercept:
            intercept_gradient = np.mean(P - Y, axis = 0)
        else:
            intercept_gradient = None

        return loss, gradient, intercept_gradient
        

    def fit(self, X, y):

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X deve ser um array de duas dimensões.")

        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("y deve ser um array de uma dimensão.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter o mesmo número de observações.")

        if X.shape[0] == 0:
            raise ValueError("X não pode estar vazio.")

        if np.any(y != y):
            raise ValueError("y não pode ter valores NaN.")

        if X.shape[1] == 0:
            raise ValueError("X deve ter pelo menos uma característica.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X deve ter apenas valores finitos.")
        
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y deve ter pelo menos duas classes.")
        
        K = len(self.classes_)
        p = X.shape[1]
        n = X.shape[0]
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.coef_ = np.zeros((p,K))
    
        if self.fit_intercept:
            self.intercept_ = np.zeros(K)

        y_indices = np.searchsorted(self.classes_, y)

        Y = np.zeros((n, K))
        Y[np.arange(n), y_indices] = 1

        # evita divisão por zero na hora de padronizar
        self.std_[self.std_ == 0] = 1

        # padroniza as variáveis de entrada
        X = (X - self.mean_) / self.std_

        self.loss_history_ = []
     
        for iter in range(self.max_iter):
            intercept = self.intercept_ if self.fit_intercept else None

            loss, gradient, intercept_gradient = self._loss_and_gradient(X, Y, self.coef_, intercept)

            # critério de convergência 
            if len(self.loss_history_) > 0:
                variation = abs(loss - self.loss_history_[-1])
                self.loss_history_.append(loss)

                if variation < self.tol:
                    break
            else:
                self.loss_history_.append(loss)

            # atualiza os coeficientes
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * intercept_gradient

            # atualiza os coeficientes
            self.coef_ -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("O modelo ainda não foi treinado. Chame fit antes.")

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X deve ser um array de duas dimensões.")

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("X deve ter o mesmo número de características usado no fit.")

        if X.shape[0] == 0:
            raise ValueError("X não pode estar vazio.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X deve ter apenas valores finitos.")
        
        X = (X - self.mean_) / self.std_

        if self.fit_intercept:
            z = X @ self.coef_ + self.intercept_
        else:
            z = X @ self.coef_

        # evita que a exponencial exploda
        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)
        prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return prob


    def predict(self, X):
        prob = self.predict_proba(X)
        predictions = np.argmax(prob, axis=1)

        return self.classes_[predictions]
            


   