"""Decision trees (CART) for regression and classification."""

import numpy as np

from dataclasses import dataclass

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class _Node:
    """
    Nó da árvore.
    
    Um nó interno tem ''feature'', ''threshold'', ''left'' e ''rignt''.
    Uma folha tem apenas''value''.
    """
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None  # só nas folhas

def _node_depth(node): # função auxiliar
    """
    Devolve a profundidade da subárvore que começa em ''node''.
    
    Parameters
    ----------
    node : _Node
        Raís da subárvore.

    Returns
    -------
    depth : int
        Número de divisões no caminho mais longo até uma folha.
        Uma folha tem profundidade 0
    """
    if node.feature is None:
        return 0
    return 1 + max(_node_depth(node.left), _node_depth(node.right))

class _BaseDecisionTree(BaseEstimator):
    """
    Classe-base das árvores de decisão
    
    Implementa a construção recursiva e a predição.
    As subclasses só definem o critério de impureza e o valor das folhas.

    Parameters
    ----------
    max_depth : int ou None, default=None
        Profundidade máxima da árvore.
        Se None, a árvore cresce até todos os nós ficarem puros ou pequenos demais para dividir.
    min_samples_split : int, default=2
        Número mínimo de amostras para tentar dividir um nó.

    Attributes
    ----------
    tree_ : _Node
        Nó raíz da árvore treinada.
    n_features_in_ : int
        Número de features vistas no "fit".
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    # ------------------------------------------------------------------
    # Validação (temporária: trocar por utils/validation quando existir)
    # ------------------------------------------------------------------
        
    def _check_params(self):
        """
        Valida os hiperparâmetros.

        Raises
        ------
        ValueError
            Se ''max_depth'' não for None nem um inteiro >= 1,
            ou se ''min_samples_split'' não for um inteiro >= 2.
        """
        if self.max_depth is not None:
            if (
                not isinstance(self.max_depth, (int, np.integer)) or
                self.max_depth < 1
            ):
                raise ValueError("max_depth must be None or an integer >= 1.")
        if (
            not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer >= 2.")

    @staticmethod
    def _check_X(X): 
        """
        Converte X para um array 2D de floats.

        Parameters
        ----------
        X : array-like
            Dados de entrada.

        Returns
        -------
        X : ndarray de shape (n_amostras, n_features)
            cópia de X como float.

        Raises
        ------
        ValueError
            Se X não for 2D.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                "X must be 2D with shape (n_samples, n_features)."
                "For a single feature, use X.reshape(-1,1).")
        return X.astype(float)

    def _check_is_fitted(self):
        """
        Confere que o modelo foi treinado.
        
        Raises
        ------
        RuntimeError
            Se ''fit'' ainda não foi chamado.
        """
        if not hasattr(self, "tree_"):
            raise RuntimeError(
                f"This {type(self).__name__} is not fitted yet. Call ''fit'' first."
            )

    def fit(self, X, y):
        # validar as entradas
        self._check_params()
        X = self._check_X(X)
        y = np.asarray(y)

        # X tem que ser 2D (n_amostras, n_features), y é 1D com o mesmo num de linhas
        # max_depth: None ou inteiro >=1
        # min_samples_split: inteiro >=2
        if y.dim != 1:
            raise ValueError("y must be 1D.")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if len(y) == 0:
            raise ValueError("Cannot fit with zero samples.")

        # Regressão: y vira array de floats        
        # Classificação: classes_ = valores únicos ordenados de y
        y = self._prepare_y(y)

        # y = índice de cada rótulo dentro de classes_
        # usar np.unique(y, return_inverse=True)
        # As impurezas trabalham com inteiros e os rótulos originais só voltam no predict

        # construir a árvore recursivamente
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        # Validar X como 2D e conferir que existe tree_ (o modelo foi treinado)
        if not hasattr(self, "tree_"):
            raise RuntimeError(f"This {type(self).__name__} is not fitted yet. Call fit first.")
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}."
            )

        values = []

        for x in X: #descer pela árvore, x é linha da matriz
            nod = self.tree_
            while nod.feature is not None: # enquanto nó não é folha
                if x[nod.feature] <= nod.threshold:
                    nod = nod.left
                else:
                    nod = nod.right
            values.append(nod.value) #checar identação desta linha

        # Classificação: value é um índice de classes, pegar classes_[indice], lembrar de converter value float -> int
        # Regressão: devolver valores como estão

        return self._decode_predictions(np.array(values))

    def _is_leaf(self, y, depth):

        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if len(y) < self.min_samples_split:
            return True

        if self._impurity(y) <= 1e-12: #nó puro
            return True

        return False  

    def _build(self, X, y, depth):
        n = len(y)

        if self._is_leaf(y, depth):
            return _Node(value= self._leaf_value(y))
        
        # Busca da melhor divisão
        best_feature, best_threshold = None, None
        best_score = np.inf #infinito

        for j in range(X.shape[1]):
        # Limiares candidatos: pontos médios entre valores consecutivos distintos e ordenados da coluna j
            valores = np.unique(X[:, j]) #já ordenados
            limiares = (valores[:, -1] + valores[1:])/2 #mudar para portugueês os nomes

            for t in limiares:
                mask = X[:, j] <= t #True = vai p esquerda
                y_esq, y_dir = y[mask], y[~mask]
                if len(y_esq) == 0 or len(y_dir) == 0:
                    continue #pode acontecer por arredondamento de floats

                score = (
                    len(y_esq)*self._impurity(y_esq)
                    + len(y_dir)*self._impurity(y_dir)
                    )/n

                if score < best_score:
                    best_score = score
                    best_feature = j
                    best_threshold = t
        
        if best_feature is None: #linhas idênticas
            return _Node(value=self._leaf_value(y))

        # recursão
        mask = X[:, best_feature] <= best_threshold
        nod = _Node(feature=best_feature, threshold=best_threshold)
        nod.left = self._build(X[mask], y[mask], depth+1)
        nod.right = self._build(X[~mask], y[~mask], depth+1)
        return nod
        
    def _prepare_y(self, y):
        pass

    def _decode_predictions(self, values):
        pass

    def _impurity(self, y):  # cada subclasse define
        pass

    def _leaf_value(self, y):  # cada subclasse define
        pass

class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def _impurity(self, y):
        def mean(y):
            result = 
            return result
        return mean((y- mean(y))**2)

    def _leaf_value(self, y):
        pass


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def _impurity(self, y):
        Gini = 1 - sum(p_k**2) #p_k é a proporção da classe k em y
        counts = np.bincount(y)
        p = counts/len(y)

        # Nó puro é Gini = 0

        return 1 - np.sum(p**2)

    def _leaf_value(self, y):
        # empate: argmax pega o menor índice
        # fit já deixou os índices inteiros

        return np.argmax(np.bincount(y))
        