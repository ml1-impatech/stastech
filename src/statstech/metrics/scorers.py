"""Registro das métricas por nome, usado por cross_val_score e GridSearchCV."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

import statstech.metrics.classification as cl
import statstech.metrics.regression as rg

# Função de scoring com assinatura scorer(y_true, y_pred) -> float.
ScorerFunc = Callable[[npt.ArrayLike, npt.ArrayLike], float]

# Qualquer scorer registrado.
AnyScorer = Callable[..., float]


def _from_confusion_matrix(
    metric: Callable[[list[list[int]]], float],
) -> ScorerFunc:
    """Adapta uma métrica de matriz de confusão para (y_true, y_pred).

    Parameters
    ----------
    metric : callable
        Função que recebe uma matriz de confusão e devolve um float
        (por exemplo, accuracy_score ou macro_f1).

    Returns
    -------
    scorer : callable
        Função com assinatura scorer(y_true, y_pred) que monta a
        matriz de confusão e aplica a métrica sobre ela. As classes
        são as que aparecem em y_true e y_pred.
    """

    def scorer(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
        true_arr = np.asarray(y_true)
        pred_arr = np.asarray(y_pred)
        classes = np.unique(np.concatenate([true_arr, pred_arr])).tolist()
        c_matrix = cl.confusion_matrix(true_arr.tolist(), pred_arr.tolist(), classes)
        return metric(c_matrix)

    return scorer


REGRESSION_SCORERS: dict[str, ScorerFunc] = {
    "rss": rg.residual_sum_squares,
    "mse": rg.mean_squared_error,
    "mae": rg.mean_absolute_error,
    "rmse": rg.root_mean_squared_error,
    "r2": rg.r2_score,
}

CLASSIFICATION_SCORERS: dict[str, AnyScorer] = {
    "accuracy": _from_confusion_matrix(cl.accuracy_score),
    "precision_macro": _from_confusion_matrix(cl.macro_precision),
    "recall_macro": _from_confusion_matrix(cl.macro_recall),
    "specificity_macro": _from_confusion_matrix(cl.macro_specificity),
    "f1_macro": _from_confusion_matrix(cl.macro_f1),
    "log_loss": cl.log_loss,
}

SCORERS = {
    "regression": REGRESSION_SCORERS,
    "classification": CLASSIFICATION_SCORERS,
}

LOWER_IS_BETTER = {"rss", "mse", "mae", "rmse", "log_loss"}

NEEDS_PROBA = {"log_loss"}

_FLAT: dict[str, AnyScorer] = {**REGRESSION_SCORERS, **CLASSIFICATION_SCORERS}


def lower_is_better(name: str) -> bool:
    """Indica se uma métrica é melhor quanto menor for.

    Parameters
    ----------
    name : str
        Nome da métrica (por exemplo, "mse" ou "accuracy").

    Returns
    -------
    bool
        True se valores menores são melhores, False caso contrário.
    """
    return name in LOWER_IS_BETTER


def needs_proba(name: str) -> bool:
    """Indica se uma métrica precisa de probabilidades (predict_proba).

    Parameters
    ----------
    name : str
        Nome da métrica (por exemplo, "log_loss" ou "accuracy").

    Returns
    -------
    bool
        True se a métrica usa probabilidades em vez das previsões.
    """
    return name in NEEDS_PROBA


def get_scorer(name: str) -> AnyScorer:
    """Devolve a função de scoring correspondente a um nome.

    Parameters
    ----------
    name : str
        Nome da métrica (por exemplo, "mse", "r2" ou "accuracy").

    Returns
    -------
    scorer : callable
        Função que devolve um float. Em geral tem assinatura
        scorer(y_true, y_pred). Se needs_proba(name) for True, a
        assinatura é outra (ex.: log_loss recebe probabilidades).

    Raises
    ------
    ValueError
        Se name não for o nome de uma métrica conhecida.
    """
    if name not in _FLAT:
        raise ValueError(f"scoring desconhecido: {name!r}. Opções: {sorted(_FLAT)}")
    return _FLAT[name]
