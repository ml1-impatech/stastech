"""Decision trees (CART) for regression and classification."""

from dataclasses import dataclass

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None  # only in leaves


class _BaseDecisionTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        

    def fit(self, X, y):
        # validar as entradas, esperar alguém implementar utils/validation
        # X tem que ser 2D (n_amostras, n_features), y é 1D com o mesmo num de linhas
        # max_depth: None ou inteiro >=1
        # min_samples_split: inteiro >=2

        # Regressão: y vira array de floats
        # Classificação: classes_ = valores únicos ordenados de y
        # y = índice de cada rótulo dentro de classes_
        # usar np.unique(y, return_inverse=True)
        # As impurezas trabalham com inteiros e os rótulos originais só voltam no predict

        # construir a árvore recursivamente
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        pass

    def _build(self, X, y, depth):
        n = len(y)

        # Critérios de parada (se algum for verdadeiro, vira folha)
        # Mudar a forma de implementação para não ficar repetindo linha
        if self.max_depth != None and depth >= self.max_depth:
            return _Node(value= self._leaf_value(y))
        if n < self.min_samples_split:
            return _Node(value= self._leaf_value(y))
        if self._impurity == 0:
            return _Node(value= self._leaf_value(y))
        
        # Busca da melhor divisão
        best_feature, best_threshold = None, None
        best_score = 10000000 #infinito

        for j in colunas de X:
        # Limiares candidatos: pontos médios entre valores consecutivos distintos e ordenados da coluna j
            valores = valores únicos de X[:, j] ordenados
            limiares = (valores[:, -1] + valores[1:])/2

            for t in limiares:
                mask = X[:, j] <= t #(True = vai p esquerda)
                y_esq, y_dir = y[mask], y[~mask]

                score = (len(y_esq)*self._impurity(y_esq) + len(y_dir)*self._impurity(y_dir))/n

                if score < best_score:
                    #guarda score, j e t como os melhores
        
        if best_feature == None:
            return _Node(value=self._leaf_value(y))

        # recursão
        mask = X[:, best_feature] <= best_threshold
        nod = _Node(feature=best_feature, threshold=best_threshold)
        nod.left = self._build(X[mask], y[mask], depth+1)
        nod.right = self._build(X[~mask], y[~mask], depth+1)
        return nod
        

    def _impurity(self, y):  # each subclass defines
        pass

    def _leaf_value(self, y):  # each subclass defines
        pass


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass