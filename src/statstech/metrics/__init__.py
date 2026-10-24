"""Métricas de regressão e de classificação."""

from statstech.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statstech.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    residual_sum_squares,
    root_mean_squared_error,
)
from statstech.metrics.scorers import get_scorer

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "get_scorer",
    "log_loss",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "residual_sum_squares",
    "roc_auc_score",
    "root_mean_squared_error",
]
