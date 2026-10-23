import statstech.metrics.regression as rg
import statstech.metrics.classification as cl

REGRESSION_SCORERS = {
    "rss": rg.residual_sum_squares,
    "mse": rg.mean_squared_error,
    "mae": rg.mean_absolute_error,
    "rmse": rg.root_mean_squared_error,
    "r2": rg.r2_score,
}

CLASSIFICATION_SCORERS = {
    "accuracy": cl.accuracy_score,
    "precision_macro": cl.macro_precision,
    "recall_macro": cl.macro_recall,
    "specificity_macro": cl.macro_specificity,
    "f1_macro": cl.macro_f1,
    "log_loss": cl.log_loss
}

SCORERS = {
    "regression": REGRESSION_SCORERS,
    "classification": CLASSIFICATION_SCORERS,
}

LOWER_IS_BETTER = {"rss", "mse", "mae", "rmse", "log_loss"}

def lower_is_better(name: str) -> bool:
    "Retorna True se uma métrica é melhor quanto menor for."
    return name in LOWER_IS_BETTER

_FLAT = {**REGRESSION_SCORERS, **CLASSIFICATION_SCORERS}

def get_scorer(name):
    if name not in _FLAT:
        raise ValueError(f"scoring desconhecido: {name!r}. Opções: {sorted(_FLAT)}")
    return _FLAT[name]