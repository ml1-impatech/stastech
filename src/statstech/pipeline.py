"""Encadeamento de transformadores seguidos de um modelo."""

from typing import Any, Self

from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, clone
from statstech.utils.validation import check_is_fitted


class Pipeline(BaseEstimator):
    """Encadeia transformadores e um modelo final.

    No fit, cada transformador é ajustado nos dados que recebe e passa
    a saída adiante; o último passo é ajustado na saída do penúltimo.
    Dentro de cross_val_score ou GridSearchCV, o pipeline é clonado e
    reajustado em cada fold, então os transformadores (por exemplo, o
    StandardScaler) só veem os dados de treino daquele fold.

    Parameters
    ----------
    steps : list de tuplas (str, objeto)
        Passos em ordem. Todos, menos o último, precisam ter fit e
        transform; o último precisa ter fit. Os nomes devem ser únicos
        e não podem conter "__".

    Attributes
    ----------
    steps_ : list de tuplas (str, objeto)
        Cópias dos passos, ajustadas no fit.
    named_steps_ : dict
        Os mesmos passos de steps_, acessíveis pelo nome.
    classes_ : ndarray de formato (n_classes,)
        Classes do último passo. Só existe se ele for um classificador.

    Notes
    -----
    Para alterar um parâmetro de um passo, use o nome do passo, dois
    sublinhados e o nome do parâmetro: set_params(model__alpha=1.0).
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        self.steps = steps

    def _check_steps(self) -> None:
        """Valida a lista de passos.

        Raises
        ------
        ValueError
            Se steps não for uma lista não vazia de tuplas (nome, objeto),
            se algum nome não for str, tiver "__" ou se repetir, se algum
            passo antes do último não tiver fit e transform ou se o último
            não tiver fit.
        """

        if not isinstance(self.steps, list) or len(self.steps) == 0:
            raise ValueError("steps deve ser uma lista não vazia.")
        for item in self.steps:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError("Cada passo deve ser uma tupla (nome, objeto).")

        names = [name for name, _ in self.steps]
        for name in names:
            if not isinstance(name, str) or "__" in name:
                raise ValueError(f"Nome de passo inválido: {name!r}.")
        if len(set(names)) != len(names):
            raise ValueError(f"Os nomes dos passos devem ser únicos: {names}.")

        for name, obj in self.steps[:-1]:
            if not (hasattr(obj, "fit") and hasattr(obj, "transform")):
                raise ValueError(f"O passo {name!r} precisa ter fit e transform.")
        if not hasattr(self.steps[-1][1], "fit"):
            raise ValueError("O último passo precisa ter fit.")

    def set_params(self, **params: Any) -> Self:
        """Altera parâmetros do pipeline ou dos seus passos.

        Um parâmetro de um passo é indicado pelo nome do passo, dois
        sublinhados e o nome do parâmetro (por exemplo, model__alpha).
        Os demais vão para o próprio pipeline (por exemplo, steps).

        Parameters
        ----------
        **params
            Parâmetros a alterar, na forma nome=valor.

        Returns
        -------
        self : Pipeline
            O próprio pipeline, com os novos valores.

        Raises
        ------
        ValueError
            Se o nome antes de "__" não for o nome de um passo, ou se o
            parâmetro não existir no passo ou no pipeline.
        """
        own_params = {}
        estimators = dict(self.steps)
        for key, value in params.items():
            if "__" not in key:
                own_params[key] = value
                continue
            step_name, param = key.split("__", 1)
            if step_name not in estimators:
                raise ValueError(
                    f"Passo '{step_name}' não existe. Passos: {list(estimators)}"
                )
            estimators[step_name].set_params(**{param: value})
        return super().set_params(**own_params)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Ajusta cada passo, em ordem.

        Cada transformador é ajustado nos dados que recebe e passa a
        saída adiante; o último passo é ajustado na saída do penúltimo.
        Os passos ajustados são cópias: steps não é alterado.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : array-like de formato (n_amostras,)
            Alvo.

        Returns
        -------
        self : Pipeline
            O próprio pipeline, ajustado.

        Raises
        ------
        ValueError
            Se steps for inválido (ver _check_steps).
        """
        self._check_steps()
        self.steps_ = [(name, clone(step)) for name, step in self.steps]
        Xt = X
        for _, transformer in self.steps_[:-1]:
            Xt = transformer.fit(Xt, y).transform(Xt)
        final = self.steps_[-1][1]
        final.fit(Xt, y)
        self.named_steps_ = dict(self.steps_)
        if hasattr(final, "classes_"):
            self.classes_ = final.classes_
        return self

    def _transform(self, X: ArrayLike) -> NDArray[Any]:
        """Passa X pelos transformadores já ajustados, sem o último passo."""
        check_is_fitted(self, "steps_")
        Xt = X
        for _, transformer in self.steps_[:-1]:
            Xt = transformer.transform(Xt)
        return Xt

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Transforma X e prevê com o último passo.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.

        Returns
        -------
        y_pred : ndarray de formato (n_amostras,)
            Previsões do último passo.

        Raises
        ------
        NotFittedError
            Se o pipeline ainda não foi ajustado.
        """

        Xt = self._transform(X)
        return self.steps_[-1][1].predict(Xt)

    def predict_proba(self, X: ArrayLike) -> NDArray[Any]:
        """Transforma X e prevê as probabilidades com o último passo.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.

        Returns
        -------
        probabilities : ndarray de formato (n_amostras, n_classes)
            Probabilidades previstas pelo último passo. As colunas seguem
            a ordem de classes_.

        Raises
        ------
        NotFittedError
            Se o pipeline ainda não foi ajustado.
        AttributeError
            Se o último passo não tiver predict_proba.
        """

        Xt = self._transform(X)
        return self.steps_[-1][1].predict_proba(Xt)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Transforma X e calcula o score do último passo.

        Usa o método score do último passo: acurácia num classificador
        e R² num regressor.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.
        y : array-like de formato (n_amostras,)
            Valores verdadeiros do alvo.

        Returns
        -------
        score : float
            Score do último passo em (X transformado, y).

        Raises
        ------
        NotFittedError
            Se o pipeline ainda não foi ajustado.
        """
        Xt = self._transform(X)
        return self.steps_[-1][1].score(Xt, y)
