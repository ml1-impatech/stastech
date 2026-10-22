"""Funções de distância usadas por vários módulos (KNN, K-means)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.utils.validation import check_array

_METRICS = ("euclidean", "manhattan")


def pairwise_distances(
    A: ArrayLike,
    B: ArrayLike | None = None,
    metric: str = "euclidean",
    squared: bool = False,
) -> NDArray[np.float64]:
    """Calcula a distância entre cada linha de A e cada linha de B.

    Parameters
    ----------
    A : array-like de formato (n_a, n_features)
        Primeiro conjunto de pontos.
    B : array-like de formato (n_b, n_features), default=None
        Segundo conjunto de pontos. Se None, usa B = A.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Tipo de distância.
    squared : bool, default=False
        Se True, devolve a distância euclidiana ao quadrado (evita a
        raiz quadrada, útil no K-means). Só vale para "euclidean".

    Returns
    -------
    distances : ndarray de formato (n_a, n_b)
        distances[i, j] é a distância entre A[i] e B[j].

    Raises
    ------
    ValueError
        Se A ou B forem inválidos, se tiverem números de features
        diferentes, se a métrica não existir ou se squared=True for
        usado com outra métrica que não a euclidiana.
    """
    if metric not in _METRICS:
        raise ValueError(f"Métrica '{metric}' inválida. Opções: {list(_METRICS)}")
    if squared and metric != "euclidean":
        raise ValueError("squared=True só vale para a métrica euclidiana.")

    A = check_array(A).astype(float)
    B = A if B is None else check_array(B).astype(float)

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            "A e B precisam ter o mesmo número de features: "
            f"{A.shape[1]} e {B.shape[1]}"
        )

    if metric == "manhattan":
        # Broadcasting: (n_a, 1, p) - (1, n_b, p) -> (n_a, n_b, p)
        diff = np.abs(A[:, None, :] - B[None, :, :])
        return diff.sum(axis=2)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (sem laço duplo)
    sq_a = np.sum(A**2, axis=1)[:, None]
    sq_b = np.sum(B**2, axis=1)[None, :]
    sq_dist = sq_a + sq_b - 2 * (A @ B.T)

    # Erro de arredondamento pode deixar valores levemente negativos.
    sq_dist = np.maximum(sq_dist, 0.0)

    if squared:
        return sq_dist
    return np.sqrt(sq_dist)
