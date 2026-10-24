"""Validação cruzada de modelos."""

from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import clone
from statstech.metrics.scorers import get_scorer, needs_proba
from statstech.model_selection.split import KFold
from statstech.utils.validation import check_X_y


class EstimatorProtocol(Protocol):
    """Estimador aceito pela validação cruzada: fit, predict e score."""

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Any: ...
    def predict(self, X: NDArray[Any]) -> NDArray[Any]: ...
    def score(self, X: NDArray[Any], y: NDArray[Any]) -> float: ...


class SplitterProtocol(Protocol):
    """Divisor de dados aceito como cv, KFold e LeaveOneOut."""

    def split(
        self, X: NDArray[Any]
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...


# Tipo do parâmetro scoring: uma função que recebe (y_true, y_pred) e retorna float
ScoringCallable = Callable[[NDArray[Any], NDArray[Any]], float]


def _fit_and_score(
    estimator: EstimatorProtocol,
    X: NDArray[Any],
    y: NDArray[Any],
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    scorer: Callable[..., float] | None = None,
    use_proba: bool = False,
) -> float:
    """Treina uma cópia do modelo e pontua no conjunto de teste.

    Clona o estimador, realiza o treino nos índices especificados
    e avalia o desempenho usando a métrica padrão ou uma dada.

    Parameters
    ----------
    estimator : objeto com fit, predict e score
        Modelo a ser avaliado.

    X : ndarray de formato (n_amostras, n_features)
        Dados de entrada (features).

    y : ndarray de formato (n_amostras,)
        Rótulos ou valores correspondentes a X.

    train_idx : array-like de int
        Índices ou máscara booleana para filtrar
        as amostras de treino em X e y.

    test_idx : array-like de int
        Índices ou máscara booleana para filtrar
        as amostras de teste em X e y.

    scorer : callable ou None
        Função com assinatura scoring(y_true, y_pred) que devolve
        um número. Se None, usa o método score do estimator.
        Se use_proba=True, a assinatura é (y_true, y_pred, classes_).

    use_proba : bool, default=False
        Se True, usa predict_proba e classes_ em vez de predict e
        scorer é (y_true, y_pred, classes_)

    Returns
    -------
    score : float
        A pontuação do modelo avaliado nos dados de teste.
    """
    model = clone(estimator)
    model.fit(X[train_idx], y[train_idx])

    X_test, y_test = X[test_idx], y[test_idx]
    if scorer is None:
        return model.score(X_test, y_test)
    if use_proba:
        return scorer(y_test, model.predict_proba(X_test), model.classes_)
    return scorer(y_test, model.predict(X_test))


def cross_val_score(
    estimator: EstimatorProtocol,
    X: ArrayLike,
    y: ArrayLike,
    cv: int | SplitterProtocol = 5,
    scoring: ScoringCallable | str | None = None,
) -> NDArray[np.float64]:
    """Avalia um modelo por validação cruzada.

    Para cada partição gerada por cv, treina uma cópia do modelo
    com os dados de treino e calcula o score nos dados de teste.
    O estimator original não é modificado nem treinado.

    Parameters
    ----------
    estimator : objeto com fit, predict e score
        Modelo a ser avaliado. Uma cópia nova é criada com clone
        a cada partição. Com scoring="log_loss", é preciso que
        tenha predict_proba e classes_

    X : array-like de formato (n_amostras, n_features)
        Dados de entrada. Deve ser 2D e numérico, sem NaN ou inf.
        Para uma única feature, use X.reshape(-1, 1).

    y : array-like de formato (n_amostras,)
        Valores alvo. Deve ser 1D e ter o mesmo número de amostras
        de X.

    cv : int ou objeto divisor, default=5
        Se int, usa KFold com esse número de partições. Também
        aceita um divisor já configurado (KFold, LeaveOneOut) que
        tenha o método split.

    scoring : callable, str ou None, default=None
        Função com assinatura scoring(y_true, y_pred) que devolve
        um número. Se None, usa o método score do estimator.
        Pode receber um nome como "mse" (ver get_scorer).

    Returns
    -------
    scores : ndarray de formato (n_partições,)
        Score obtido no conjunto de teste de cada partição.

    Raises
    ------
    ValueError
        Se X ou y forem inválidos (ver check_X_y): X que não seja 2D
        ou numérico, y que não seja 1D, valores não finitos ou
        tamanhos diferentes.
        Se scoring="log_loss" e um fold de teste tiver uma classe que
        não apareceu no treino daquele fold.
    TypeError
        Se scoring="log_loss" e estimator não tiver predict_proba.
        Se scoring não for None, um nome desconhecido (str) ou uma função.
    """
    scorer: Callable[..., float] | None
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    elif scoring is None or callable(scoring):
        scorer = scoring
    else:
        raise TypeError("scoring deve ser None, um nome (str) ou uma função.")

    use_proba = isinstance(scoring, str) and needs_proba(scoring)
    if use_proba and not hasattr(estimator, "predict_proba"):
        raise TypeError(f"scoring={scoring!r} exige um estimador com predict_proba.")

    X_arr, y_arr = check_X_y(X, y)

    splitter = KFold(n_splits=cv) if isinstance(cv, int) else cv

    return np.array(
        [
            _fit_and_score(
                estimator, X_arr, y_arr, train_idx, test_idx, scorer, use_proba
            )
            for train_idx, test_idx in splitter.split(X_arr)
        ]
    )
