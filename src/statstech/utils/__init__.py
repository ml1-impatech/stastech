"""Funções de validação e de distância usadas por toda a biblioteca."""

from statstech.utils.distances import pairwise_distances
from statstech.utils.validation import (
    NotFittedError,
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
)

__all__ = [
    "NotFittedError",
    "check_X_y",
    "check_array",
    "check_is_fitted",
    "check_n_features",
    "pairwise_distances",
]