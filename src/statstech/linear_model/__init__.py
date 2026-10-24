"""Modelos lineares: mínimos quadrados, Ridge, Lasso e regressão logística."""

from statstech.linear_model.lasso import Lasso
from statstech.linear_model.linear_regression import LinearRegression
from statstech.linear_model.logistic_regression import LogisticRegression
from statstech.linear_model.ridge import Ridge

__all__ = ["Lasso", "LinearRegression", "LogisticRegression", "Ridge"]
