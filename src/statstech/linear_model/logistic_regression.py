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

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_binary = []
        n = X.shape[0]
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # transforma os rótulos em 0 e 1 (assumindo que é binário)
        for element in y:
            if element == self.classes_[1]:
                y_binary.append(1)
            else:
                y_binary.append(0)

        y_binary = np.array(y_binary)
        self.coef_ = np.zeros(X.shape[1])

        if self.fit_intercept == True:
            self.intercept_ = 0 

        # padroniza as variáveis de entrada
        for j in range(len(self.std_)):
            if self.std_[j] == 0:
                self.std_[j] = 1
            
        X = (X - self.mean_) / self.std_
        self.loss_history_ = []
     
        for i in range(self.max_iter):

            if self.fit_intercept == True:
                z = X @ self.coef_ + self.intercept_
            else:
                z = X @ self.coef_

            p = np.zeros(len(z))
            
            # sigmoide numericamente estável
            for j in range(len(z)):
                if z[j] >= 0:
                    p[j] = 1 / (1 + np.exp(-z[j]))
                else:
                    p[j] = np.exp(z[j]) / (1 + np.exp(z[j]))

            # evita logaritmo de zero
            p = np.clip(p, 1e-15, 1 - 1e-15)
            loss = -np.mean( y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p) )

            if self.lambda_ > 0:
                L2 = (self.lambda_ / (2*n)) * np.sum(self.coef_ ** 2)
                loss = loss + L2

            if len(self.loss_history_) > 0:
                        variation = abs(loss - self.loss_history_[-1])
                        self.loss_history_.append(loss)

                        if variation < self.tol:
                            break
            else:
                self.loss_history_.append(loss)

            gradient = ( X.T @ (p - y_binary) )/n 

            # regularização
            if self.lambda_ > 0:
                gradient = gradient + (self.lambda_ / n) * self.coef_
            
            if self.fit_intercept == True:
                intercept_gradient = np.mean(p - y_binary)
                self.intercept_ = self.intercept_ - self.learning_rate * intercept_gradient

            new_coef = self.coef_ - self.learning_rate * gradient
            self.coef_ = new_coef

        return self
         

  


            