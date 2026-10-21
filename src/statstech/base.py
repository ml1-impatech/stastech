"""Classes e funções base usadas por todos os modelos da biblioteca."""

import copy
import inspect
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from statstech.metrics.regression import r2_score


class _HasPredict(Protocol):
    """Objeto com o método predict (usado pelos mixins de score)."""

    def predict(self, X: NDArray[Any]) -> NDArray[Any]: ...


class _HasFitTransform(Protocol):
    """Objeto com os métodos fit e transform."""

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> Any: ...

    def transform(self, X: NDArray[Any]) -> NDArray[Any]: ...


class BaseEstimator:
    """Classe base de todos os modelos e transformadores.

    Exige que o __init__ guarde cada parâmetro recebido em um atributo
    com o mesmo nome (por exemplo, self.alpha = alpha).
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Devolve os nomes dos parâmetros do __init__.

        Returns
        -------
        names : list of str
            Nomes dos parâmetros, em ordem alfabética (sem self).

        Raises
        ------
        RuntimeError
            Se o __init__ usar *args.
        """

        init = cls.__init__

        if init is object.__init__:
            return []

        signature = inspect.signature(init)
        names = []

        for p in signature.parameters.values():
            if p.name == "self" or p.kind == p.VAR_KEYWORD:
                continue
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(f"{cls.__name__}: __init__ não pode usar *args.")
            names.append(p.name)

        return sorted(names)

    def get_params(self) -> dict[str, Any]:
        """Devolve os hiperparâmetros do modelo.

        Returns
        -------
        params : dict
            Dicionário {nome_do_hiperparâmetro: valor}.
        """

        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> Self:
        """Altera hiperparâmetros do modelo.

        Parameters
        ----------
        **params
            Hiperparâmetros a alterar, na forma nome=valor.

        Returns
        -------
        self : o próprio modelo, com os novos valores.

        Raises
        ------
        ValueError
            Se algum nome não for um parâmetro válido do modelo.
        """

        valid_params = self._get_param_names()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Parâmetro inválido '{key}' para o estimador "
                    f"{type(self).__name__}. "
                    f"Parâmetros válidos são: {list(valid_params)}"
                )
            setattr(self, key, value)

        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class ClassifierMixin:
    """Mixin para classificadores: acrescenta o método score."""

    def score(self: _HasPredict, X: NDArray[Any], y: NDArray[Any]) -> float:
        """Calcula a acurácia do modelo.

        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Dados de entrada.
        y : ndarray de shape (n_amostras,)
            Classes verdadeiras.

        Returns
        -------
        score : float
            Proporção de classes previstas corretamente.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class RegressorMixin:
    """Mixin para regressores: acrescenta o método score."""

    def score(self: _HasPredict, X: NDArray[Any], y: NDArray[Any]) -> float:
        """Calcula o coeficiente de determinação R² do modelo.

        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Dados de entrada.
        y : ndarray de shape (n_amostras,)
            Valores reais do alvo.

        Returns
        -------
        score : float
            O R² das previsões em relação a y.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class TransformerMixin:
    """Mixin para transformadores: acrescenta o fit_transform."""

    def fit_transform(
        self: _HasFitTransform, X: NDArray[Any], y: NDArray[Any] | None = None
    ) -> NDArray[Any]:
        """Ajusta o transformador e devolve os dados transformados.

        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Dados de entrada.
        y : ndarray de shape (n_amostras,), default=None
            Alvo, usado apenas por transformadores supervisionados.

        Returns
        -------
        X_new : ndarray
            Dados transformados.
        """
        return self.fit(X, y).transform(X)


def clone(estimator: Any) -> Any:
    """Cria um novo modelo, sem treino, com os mesmos hiperparâmetros.

    Parameters
    ----------
    estimator : objeto com get_params, ou lista/tupla de objetos
        Modelo a ser copiado. Listas e tuplas são clonadas elemento
        por elemento.

    Returns
    -------
    new_estimator : mesmo tipo de estimator
        Novo objeto, com os hiperparâmetros copiados e sem ajuste.

    Raises
    ------
    TypeError
        Se estimator não tiver o método get_params.
    """

    if isinstance(estimator, (list, tuple)):
        return type(estimator)(clone(e) for e in estimator)

    if not hasattr(estimator, "get_params"):
        raise TypeError(f"Não foi possível clonar {estimator!r}: sem get_params.")

    params = {
        name: clone(value) if hasattr(value, "get_params") else copy.deepcopy(value)
        for name, value in estimator.get_params().items()
    }

    return type(estimator)(**params)
