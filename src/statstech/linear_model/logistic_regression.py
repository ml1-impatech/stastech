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

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.lambda_ = lambda_

    # função auxiliar para facilitar os testes do gradiente 
    def _loss_and_gradient(self, X, Y, coef, intercept):
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
        self.classes_ = np.unique(y)
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
        X = (X - self.mean_) / self.std_

        if self.fit_intercept == True:
            z = X @ self.coef_ + self.intercept_
        else:
            z = X @ self.coef_

        # evita que a exponencial exploda
        for i in range(len(z)):
            bigger = z[i, 0]
            for j in range(1, len(self.classes_)):
                if z[i,j] > bigger:
                   bigger = z[i, j]

            for j in range(len(self.classes_)):
                z[i, j] = z[i, j] - bigger

        exp_z = np.exp(z)
        prob = np.zeros(exp_z.shape)

        for i in range(len(exp_z)):
            total = 0

            for j in range(len(self.classes_)):
                total += exp_z[i, j]

            for j in range(len(self.classes_)):
                prob[i, j] = exp_z[i, j] / total

        return prob


    def predict(self, X):
        prob = self.predict_proba(X)
        predictions = np.zeros(len(X), dtype = int)

        for i in range(len(X)):
            bigger = prob[i, 0]
            index = 0 

            for j in range(1, len(self.classes_)):
                if prob[i, j] > bigger:
                    bigger = prob[i, j]
                    index = j 

            predictions[i] = index

        return self.classes_[predictions]
            


   