"""Exemplo de regressão linear, ridge, lasso e regressão polinomial
utilizando, respectivamnete, as classes LinearRegression, Ridge,
Lasso e PolynomialFeatures"""

import numpy as np

# Dados
X = np.arange(1, 11, dtype=float).reshape(-1, 1)
y = np.array([3.1, 4.8, 7.2, 8.9, 11.3, 12.8, 15.2, 16.9, 19.1, 21.2])

X_new = np.array([[2.5], [5.5], [8.5]])

# Exemplo de uso - LinearRegression
from statstech.linear_model.linear_regression import LinearRegression

# Cria e ajusta o modelo.
model = LinearRegression()
model.fit(X, y)

# Realiza previsões para novos valores.
pred = model.predict(X_new)

print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
print("Previsões:", pred)


# Exemplo de uso - Ridge
from statstech.linear_model.ridge import Ridge

# alpha controla a intensidade da regularização - não negativo
# Cria e ajusta o modelo
model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X, y)

# Realiza previsões do modelo
pred_ridge = model_ridge.predict(X_new)

print("Coeficientes do Ridge:", model_ridge.coef_)
print("Intercepto do Ridge:", model_ridge.intercept_)
print("Previsões do Ridge:", pred_ridge)

# Exemplo de uso - Lasso
from statstech.linear_model.lasso import Lasso

# alpha controla a intensidade da regularização - não negativo
# Cria e ajusta o modelo
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X, y)

# Realiza previsões do modelo
pred_lasso = model_lasso.predict(X_new)

print("Coeficientes do Lasso:", model_lasso.coef_)
print("Intercepto do Lasso:", model_lasso.intercept_)
print("Previsões do Lasso:", pred_lasso)

# Exemplo de uso - PolinomialFeatures
from statstech.preprocessing.polynomial import PolynomialFeatures

# Gera características polinomiais de grau 2
transformer = PolynomialFeatures(degree=2, include_bias=False)
X_poly = transformer.fit_transform(X)

print("Entradas originais:")
print(X)

print("Entradas transformadas:")
print(X_poly)

# Regressão polinomial
# Importar PolinomialFeatures e LinearRegression

# Cria e ajusta o modelo com as entradas de características polinomiais
model_pol_reg = LinearRegression()
model_pol_reg.fit(X_poly, y)

# Realiza previsões do modelo
X_new_poly = transformer.transform(X_new)
y_pred_pol_reg = model_pol_reg.predict(X_new_poly)

print("Coeficientes da Regressão Polinomial:", model_pol_reg.coef_)
print("Intercepto da Regressão Polinomial:", model_pol_reg.intercept_)
print("Previsões da Regressão Polinomial:", y_pred_pol_reg)
