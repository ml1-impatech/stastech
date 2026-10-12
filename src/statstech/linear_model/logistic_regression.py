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
        K = Y.shape[1]
        
        if self.fit_intercept == True:
            z = X @ coef + intercept
        else:
            z = X @ coef

        for i in range(n):
            maior = z[i,0]

            for j in range(1, K):
                if z[i,j] > maior:
                    maior = z[i,j]

            for j in range(K):
                z[i,j] = z[i,j] - maior

        exp_z = np.exp(z)
        P = np.zeros((n, K))

        for i in range(n):
            total = 0 

            for j in range(K):
                total = total + exp_z[i,j]

            for j in range(K):
                P[i,j] = exp_z[i,j] / total

        P = np.clip(P, 1e-15, 1.0)

        loss = 0

        for i in range(n):
            for j in range(K):
                loss = loss - Y[i,j] * np.log(P[i,j])

        loss = loss/n

        if self.lambda_ > 0:
            L2 = (self.lambda_ / (2*n)) * np.sum(coef ** 2)
            loss = loss + L2

        gradient = ( X.T @ (P - Y) )/n 
    
        if self.lambda_ > 0:
            gradient = gradient + (self.lambda_ / n) * coef
        
        if self.fit_intercept == True:
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
    
        if self.fit_intercept == True:
            self.intercept_ = np.zeros(K)

        y_indices = np.zeros(len(y), dtype = int)

        for i in range(len(y)):
            for j in range(K):
                if y[i] == self.classes_[j]:
                    y_indices[i] = j
                    break

        Y = np.zeros((n, K))

        for i in range(n):
            Y[i, y_indices[i]] = 1  

        # evita divisão por zero na hora de padronizar
        for j in range(p):
            if self.std_[j] == 0:
                self.std_[j] = 1
        
        # padroniza as variáveis de entrada
        X = (X - self.mean_) / self.std_

        self.loss_history_ = []
     
        for iter in range(self.max_iter):
            if self.fit_intercept:
                intercept = self.intercept_
            else:
                intercept = None

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
            if self.fit_intercept == True:
                self.intercept_ = (self.intercept_ - self.learning_rate * intercept_gradient)

            # atualiza os coeficientes
            new_coef = self.coef_ - self.learning_rate * gradient
            self.coef_ = new_coef

        return self

   