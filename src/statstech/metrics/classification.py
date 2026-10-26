"""Métricas de classificação: matriz de confusão, acurácia, precisão,
revocação, especificidade, F1, log loss e área sob a curva ROC."""

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def _validate_confusion_matrix(c_matrix: list[list[int]]) -> None:
    """Valida se a matriz de confusão é não vazia e quadrada."""
    if len(c_matrix) == 0:
        raise ValueError("A matriz de confusão deve ter ao menos um elemento.")

    if not all(len(row) == len(c_matrix) for row in c_matrix):
        raise ValueError("A matriz de confusão deve ser quadrada.")


def _validate_class_index(c_matrix: list[list[int]], class_k: int) -> None:
    """Valida a matriz de confusão e o índice da classe."""
    _validate_confusion_matrix(c_matrix)

    if isinstance(class_k, (bool, np.bool_)) or not isinstance(
        class_k, (int, np.integer)
    ):
        raise ValueError("class_k deve ser um inteiro.")

    if not 0 <= class_k < len(c_matrix):
        raise ValueError("class_k está fora do intervalo válido.")


def confusion_matrix(
    y_true: list[int], y_pred: list[int], classes: list
) -> list[list[int]]:
    """Calcula a matriz de confusão para um problema de classificação.

    A matriz de confusão compara as classes reais com as classes previstas.
    Cada linha representa uma classe verdadeira, e cada coluna uma classe prevista.

    Parameters
    ----------
    y_true: list[float]
        Lista dos valores reais correspondentes às previsões em y_pred.
    y_pred: list[float]
        Lista de predições baseadas nas observações.
    classes: list
        A lista de possíveis classes do problema de classificação.

    Returns
    -------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Raises
    ------
    ValueError
        Se y_true e y_pred tiverem comprimentos diferentes ou se algum dos dois for vazio.
        Se classes for vazio ou se houver duplicatas em classes.
    KeyError
        Se algum valor de y_true ou y_pred não estiver presente em classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "A lista de predições e de valores reais devem ter o mesmo comprimento."
        )
    if len(y_true) == 0:
        raise ValueError("As listas de input devem ser não vazias.")
    n_classes = len(classes)
    dict_classes = {}
    if len(classes) == 0:
        raise ValueError("classes deve ser não vazio.")
    for i in range(n_classes):
        dict_classes[classes[i]] = i
    c_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    if len(set(classes)) != len(classes):
        raise ValueError("classes não pode conter duplicatas.")
    for i in range(len(y_true)):
        c_matrix[dict_classes[y_true[i]]][dict_classes[y_pred[i]]] += 1
    return c_matrix


def true_positives(c_matrix: list[list[int]], class_k: int) -> int:
    """Calcula o True Positives de uma matriz de confusão para uma classe específica.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int
        O índice da classe para qual o True Positives será calculado.

    Returns
    -------
    true_positives: int
        O elemento c_matrix[class_k][class_k], o qual indica quantas vezes o classificador
        retornou class_k quando esta era a verdadeira classe da observação.

    Raises
    ------
    ValueError
        Se c_class for menor que zero ou maior que o número total de classes.
        Se a matriz de confusão não for quadrada ou for vazia.
        Se class_k for inválido.
    """
    _validate_class_index(c_matrix, class_k)

    t_positives = c_matrix[class_k][class_k]
    return t_positives


def false_positives(c_matrix: list[list[int]], class_k: int) -> int:
    """Calcula o False Positives de uma matriz de confusão para uma classe específica.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int
        O índice da classe para qual o False Positives será calculado.

    Returns
    -------
    false_positives: int
        A soma das entradas da coluna class_k, exceto c_matri[class_k][class_k].
        Isto representa o número de observações previstas como class_k quando sua
        classe verdadeira era diferente de class_k.

    Raises
    ------
    ValueError
        Se c_class for menor que zero ou maior que o número total de classes.
        Se a matriz de confusão não for quadrada ou for vazia.
        Se class_k for inválido.
    """
    _validate_class_index(c_matrix, class_k)

    f_positives = 0
    for i in range(len(c_matrix)):
        if i != class_k:
            f_positives += c_matrix[i][class_k]
    return f_positives


def true_negatives(c_matrix: list[list[int]], class_k: int) -> int:
    """Calcula o True Negatives de uma matriz de confusão para uma classe específica.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int
        O índice da classe para qual o True Negatives será calculado.

    Returns
    -------
    true_negatives: int
        Soma de todas as entradas fora da coluna class_k e linha class_k. Isto representa
        o número de observações as quais a verdadeira classe e a classe prevista são
        ambas diferentes de class_k.

    Raises
    ------
    ValueError
        Se c_class for menor que zero ou maior que o número total de classes.
        Se a matriz de confusão não for quadrada ou for vazia.
        Se class_k for inválido.
    """
    _validate_class_index(c_matrix, class_k)

    t_negatives = 0
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            if i != class_k and j != class_k:
                t_negatives += c_matrix[i][j]
    return t_negatives


def false_negatives(c_matrix: list[list[int]], class_k: int) -> int:
    """Calcula o False Negatives de uma matriz de confusão para uma classe específica.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int
        O índice da classe para qual o False Negatives será calculado.

    Returns
    -------
    false_negatives: int
        Soma de todas as entradas na linha class_k, exceto c_matrix[class_k][class_k].
        Isto representa o número de observações os quais a verdadeira classe é class_k,
        mas que foram predivistas em classes diferentes de class_k.

    Raises
    ------
    ValueError
        Se c_class for menor que zero ou maior que o número total de classes.
        Se a matriz de confusão não for quadrada ou for vazia.
        Se class_k for inválido.
    """
    _validate_class_index(c_matrix, class_k)

    f_negatives = 0
    for j in range(len(c_matrix)):
        if j != class_k:
            f_negatives += c_matrix[class_k][j]
    return f_negatives


def accuracy_score(c_matrix: list[list[int]]) -> float:
    """Calcula o Accuracy Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Returns
    -------
    true_predictions/total: float
        A razão entre as previsões certas e todas as previsões. Isto indica o quanto um
        classificador acerta em geral quando faz uma previsão.

    Raises
    ------
    ValueError
        Se não houver elementos na matriz de confusão, tornando a métrica indefinida.
        Se a matriz for completamente zerada, tornando a métrica indefinida.
        Se a matriz de confusão não for quadrada.
    """
    _validate_confusion_matrix(c_matrix)

    true_predictions = 0
    total = 0
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            if i == j:
                true_predictions += c_matrix[i][j]
            total += c_matrix[i][j]
    if total == 0:
        raise ValueError(
            "A matriz de confusão deve ter ao menos um elemento diferente de zero."
        )
    return true_predictions / total


def precision_score(c_matrix: list[list[int]], class_k: int | None = None) -> float:
    """Calcula o Precision Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int | None
        O índice da classe para qual o Precision será calculado. O padrão valor padrão
        de class_k é None, e para tal caso, o Precision é calculado através de seu macro.

    Returns
    -------
    float
        Se class_k não for None, retorna tp/(tp+fp), a proporção das observações
        previstas como pertencentes a class_k que realmente pertencem a essa classe.
        Se class_k for None, retorna macro_precision, a média aritmética dos valores
        de Precision de todas as classes.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou não quadrada, se class_k for inválido
        ou, no cálculo macro, se nenhuma classe puder ser avaliada
        devido a entradas inválidas.

    Warns
    -----
    UserWarning
        Se tp + fp for igual a zero, retornando 0.0.
    """
    if class_k == None:
        return macro_precision(c_matrix)
    tp = true_positives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    if tp + fp != 0:
        return tp / (tp + fp)
    else:
        warnings.warn(
            "Precision indefinida: tp + fp é zero; retornando 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0


def recall_score(c_matrix: list[list[int]], class_k: int | None = None) -> float:
    """Calcula o Recall Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int | None
        O índice da classe para qual o Recall será calculado. O padrão valor padrão
        de class_k é None, e para tal caso, o Recall é calculado através de seu macro.

    Returns
    -------
    float
        Se class_k não for None, retorna tp/(tp+fn), a proporção das observações que
        realmente pertencem a class_k e que foram corretamente identificadas pelo
        classificador. Se class_k for None, retorna macro_recall, a média aritmética
        dos valores de Recall de todas as classes.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou não quadrada, se class_k for inválido
        ou, no cálculo macro, se nenhuma classe puder ser avaliada
        devido a entradas inválidas.

    Warns
    -----
    UserWarning
        Se tp + fn for igual a zero, retornando 0.0.
    """
    if class_k == None:
        return macro_recall(c_matrix)
    tp = true_positives(c_matrix, class_k)
    fn = false_negatives(c_matrix, class_k)
    if tp + fn != 0:
        return tp / (tp + fn)
    else:
        warnings.warn(
            "Recall indefinido: tp + fn é zero; retornando 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0


def specificity_score(c_matrix: list[list[int]], class_k: int | None = None) -> float:
    """Calcula o Specificity Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int | None
        O índice da classe para qual o Specificity score será calculado. O padrão valor padrão
        de class_k é None, e para tal caso, o Specificity é calculado através de seu macro.

    Returns
    -------
    float
        Se class_k não for None, retorna tn/(tn+fp), a proporção das observações
        que não pertencem a class_k e que foram corretamente identificadas como
        negativas. Se class_k for None, retorna macro_specificity, a média
        aritmética dos valores de Specificity de todas as classes.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou não quadrada, se class_k for inválido
        ou, no cálculo macro, se nenhuma classe puder ser avaliada
        devido a entradas inválidas.

    Warns
    -----
    UserWarning
        Se tn + fp for igual a zero, retornando 0.0.
    """
    if class_k == None:
        return macro_specificity(c_matrix)
    tn = true_negatives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    if tn + fp != 0:
        return tn / (tn + fp)
    else:
        warnings.warn(
            "Specificity indefinida: tn + fp é zero; retornando 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0


def f1_score(c_matrix: list[list[int]], class_k: int | None = None) -> float:
    """Calcula o F1 Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.
    class_k: int | None
        O índice da classe para qual o f1 score será calculado. O padrão valor padrão
        de class_k é None, e para tal caso, o f1 é calculado através de seu macro.

    Returns
    -------
    float
        Se class_k não for None, retorna 2*tp/(2*tp+fn+fp), o F1 Score da classe
        especificada, calculado como a média harmônica entre Precision e Recall.
        Se class_k for None, retorna macro_f1, a média aritmética dos valores de
        F1 Score de todas as classes.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou não quadrada, se class_k for inválido
        ou, no cálculo macro, se nenhuma classe puder ser avaliada
        devido a entradas inválidas.

    Warns
    -----
    UserWarning
        Se o denominador do f1 for igual a zero, retornando 0.0.
    """
    if class_k == None:
        return macro_f1(c_matrix)
    tp = true_positives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    fn = false_negatives(c_matrix, class_k)
    if 2 * tp + fn + fp != 0:
        return 2 * tp / (2 * tp + fn + fp)
    else:
        warnings.warn(
            "F1 indefinido: o denominador é zero; retornando 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0


def macro_metric(
    c_matrix: list[list[int]], metric: Callable[[list[list[int]], int], float]
) -> float:
    """Calcula a média macro para uma métrica específica.

    Parameters
    ----------
    c_matrix : list[list[int]]
        Matriz de confusão quadrada.
    metric : Callable
        Métrica que será utilizada para calcular a média macro.

    Returns
    -------
    float
        Média aritmética dos valores retornados pela métrica para cada classe.
        Se a métrica lançar ValueError para uma classe, essa classe será ignorada.
        Valores 0.0 retornados pela métrica são incluídos na média.

    Raises
    ------
    ValueError
        Se nenhuma classe possuir métrica definida.

    Warns
    -----
    UserWarning
        Se uma ou mais classes forem ignoradas por terem provocado
        ValueError durante o cálculo.
    """
    total_metric = 0.0
    n_valid = 0
    n_classes = len(c_matrix)
    ignored_classes = 0

    for i in range(n_classes):
        try:
            total_metric += metric(c_matrix, i)
            n_valid += 1
        except ValueError:
            ignored_classes += 1

    if n_valid == 0:
        raise ValueError("Nenhuma classe possui métrica definida.")

    if ignored_classes > 0:
        warnings.warn(
            f"{ignored_classes} classe(s) desconsiderada(s) "
            "porque a métrica lançou ValueError.",
            UserWarning,
            stacklevel=2,
        )

    return total_metric / n_valid


def macro_f1(c_matrix: list[list[int]]) -> float:
    """Calcula a média macro do F1 Score.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Returns
    -------
    float
        O macro F1 Score, que é a média aritmética do f1_score para todas as classes.
        Classes cuja métrica apresenta denominador zero recebem valor 0.0, acompanhado
        de UserWarning. Esse valor é incluído na média macro.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou inválida, impedindo a avaliação
        de todas as classes.

    Warns
    -----
    UserWarning
        Se a métrica for indefinida para uma ou mais classes, que receberão
        valor 0.0 incluído na média macro.
    """
    return macro_metric(c_matrix, f1_score)


def macro_specificity(c_matrix: list[list[int]]) -> float:
    """Calcula a média macro do Specificity Score.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Returns
    -------
    float
        O macro Specificity Score, que é a média aritmética do specificity_score para todas as classes.
        Classes cuja métrica apresenta denominador zero recebem valor 0.0, acompanhado de
        UserWarning. Esse valor é incluído na média macro.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou inválida, impedindo a avaliação
        de todas as classes.

    Warns
    -----
    UserWarning
        Se a métrica for indefinida para uma ou mais classes, que receberão
        valor 0.0 incluído na média macro.
    """
    return macro_metric(c_matrix, specificity_score)


def macro_recall(c_matrix: list[list[int]]) -> float:
    """Calcula a média macro do Recall Score.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Returns
    -------
    float
        O macro Recall Score, que é a média aritmética do recall_score para todas as classes.
        Classes cuja métrica apresenta denominador zero recebem valor 0.0, acompanhado de
        UserWarning. Esse valor é incluído na média macro.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou inválida, impedindo a avaliação
        de todas as classes.

    Warns
    -----
    UserWarning
        Se a métrica for indefinida para uma ou mais classes, que receberão
        valor 0.0 incluído na média macro.
    """
    return macro_metric(c_matrix, recall_score)


def macro_precision(c_matrix: list[list[int]]) -> float:
    """Calcula a média macro do Precision Score.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.

    Returns
    -------
    float
        O macro Precision Score, que é a média aritmética do precision_score para todas as classes.
        Classes cuja métrica apresenta denominador zero recebem valor 0.0, acompanhado de
        UserWarning. Esse valor é incluído na média macro.

    Raises
    ------
    ValueError
        Se c_matrix for vazia ou inválida, impedindo a avaliação
        de todas as classes.

    Warns
    -----
    UserWarning
        Se a métrica for indefinida para uma ou mais classes, que receberão
        valor 0.0 incluído na média macro.
    """
    return macro_metric(c_matrix, precision_score)


def log_loss(
    y_true: npt.ArrayLike,
    p_matrix: npt.ArrayLike,
    classes: list | None = None,
    epsilon: float = 1e-10,
) -> float:
    """Calcula a perda logarítmica de um classificador.

    Parameters
    ----------
    y_true: array-like
        Array unidimensional com os rótulos reais das observações, no qual
        cada rótulo deve estar presente na lista classes.
    p_matrix: array-like
        Uma matriz de dimensões (n_samples, n_classes), na qual cada elemento
        na posição [i][j] representa a probabilidade de a observação i ser
        prevista como classe j.
    classes: list | None
        Lista de classes possíveis para se classificar. O valor padrão é None, e em tal caso,
        considera-se como possíveis classes os números inteiros entre 0 e n_classes - 1,
        onde n_classes é o número de colunas de p_matrix.
    epsilon: float, default 1e-10
        Número utilizado para evitar que se tenha log de zero.

    Returns
    -------
    l_loss: float
        Log loss média das observações. Valores menores indicam probabilidades
        mais favoráveis às classes verdadeiras.

    Raises
    ------
    ValueError
        Se as dimensões ou os valores de entrada forem inválidos.
        Se y_true contiver classes que não estão em classes, se houver
        probabilidades inválidas ou se o valor de epsilon for inválido.
    """
    p_matrix = np.asarray(p_matrix, np.float64)
    y_true = np.asarray(y_true)

    if p_matrix.ndim != 2:
        raise ValueError("A dimensão de p_matrix deve ser igual a 2.")
    if y_true.ndim != 1:
        raise ValueError("A dimensão de y_true deve ser igual a 1.")

    n_samples, n_classes = p_matrix.shape

    if n_samples == 0 or n_classes == 0:
        raise ValueError("p_matrix deve conter ao menos uma observação e uma classe.")

    if classes is not None:
        if len(classes) != n_classes:
            raise ValueError(
                "O número de classes deve ser igual ao número de colunas de p_matrix."
            )
        if len(set(classes)) != len(classes):
            raise ValueError("classes não pode conter duplicatas.")

    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon deve ser finito e estar entre 0 e 1.")
    if len(y_true) != n_samples:
        raise ValueError("As dimensões de p_matrix e y_true devem ser compatíveis.")
    if not np.all(np.isfinite(p_matrix)):
        raise ValueError("p_matrix não pode conter NaN ou infinito.")
    if np.any(p_matrix < 0) or np.any(p_matrix > 1):
        raise ValueError("p_matrix contém probabilidades fora do intervalo.")
    if not np.allclose(p_matrix.sum(axis=1), 1.0):
        raise ValueError("As probabilidades de cada linha devem somar 1.")

    if classes is None:
        dict_classes = {i: i for i in range(n_classes)}
    else:
        dict_classes = {}
        for i in range(n_classes):
            dict_classes[classes[i]] = i

    total = 0.0

    for i in range(len(y_true)):
        if y_true[i] not in dict_classes:
            raise ValueError(
                "y_true contém classes que não estão presentes em classes."
            )
        ind = dict_classes[y_true[i]]
        prob = np.clip(p_matrix[i, ind], epsilon, 1.0)
        total += np.log(prob)

    l_loss = -total / len(y_true)
    return float(l_loss)


def binary_roc_auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Calcula a área sob a curva ROC para classificação binária.

    Parameters
    ----------
    y_true: array-like
        Array unidimensional com os rótulos reais das observações,
        contendo apenas 0 e 1.
    y_score: array-like
        Array unidimensional com os scores atribuídos pelo modelo à
        classe positiva. Scores maiores indicam maior evidência da
        classe positiva.

    Returns
    -------
    auc: float
        Área sob a curva ROC, variando entre 0 e 1. Valores maiores
        indicam melhor capacidade de discriminação entre as classes.

    Raises
    ------
    ValueError
        Se os arrays não forem unidimensionais, tiverem tamanhos
        diferentes ou estiverem vazios.
        Se y_true contiver rótulos diferentes de 0 e 1, se y_score
        contiver valores não finitos ou se y_true não contiver
        ambas as classes.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true e y_score devem ser unidimensionais.")

    if y_true.size != y_score.size:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")

    if y_true.size == 0:
        raise ValueError("y_true e y_score devem ser não vazios.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true deve conter apenas 0 e 1.")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score deve conter apenas valores finitos.")

    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    if n_positive == 0 or n_negative == 0:
        raise ValueError("y_true deve conter ambas as classes.")

    order = np.argsort(-y_score)

    y_true = y_true[order]
    y_score = y_score[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    score_diffs = np.diff(y_score)

    change_positions = np.where(score_diffs != 0)[0]

    last_ind = y_true.size - 1
    distinct = np.append(change_positions, last_ind)

    tp = tp[distinct]
    fp = fp[distinct]

    tpr = tp / n_positive
    fpr = fp / n_negative

    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)

    auc = np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]) / 2.0)

    return float(auc)


def roc_auc_score(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, class_k: int | None = None
) -> float:
    """Calcula a área sob a curva ROC para classificação binária
    ou para uma classe específica em classificação multiclasse.

    Parameters
    ----------
    y_true: array-like
        Array unidimensional com os rótulos reais das observações.
        Na classificação binária, deve conter apenas 0 e 1.
        Na classificação multiclasse, deve conter índices inteiros
        de classes entre 0 e n_classes - 1.
    y_score: array-like
        Na classificação binária, array unidimensional com os scores
        da classe positiva.
        Na classificação multiclasse, matriz de dimensões
        (n_samples, n_classes), na qual cada elemento [i][j]
        representa o score atribuído à classe j para a observação i.
        A coluna j deve corresponder ao rótulo j.
    class_k: int or None, default None
        Índice da classe cuja área ROC-AUC será calculada na
        classificação multiclasse, considerando essa classe como
        positiva e todas as demais como negativas (One-vs-Rest).
        Deve ser None na classificação binária e um índice válido
        na classificação multiclasse.

    Returns
    -------
    auc: float
        Área sob a curva ROC para a classificação binária ou para
        a classe escolhida na classificação multiclasse. Valores
        maiores indicam melhor capacidade de discriminação.

    Raises
    ------
    ValueError
        Se as dimensões ou os tamanhos dos arrays forem inválidos,
        se y_score contiver valores não finitos ou se houver menos
        de duas colunas na matriz multiclasse.
        Se class_k for incompatível com o tipo de classificação,
        não for um inteiro válido ou estiver fora do intervalo das
        classes.
        Se y_true contiver rótulos que não correspondam aos índices
        das colunas de y_score na classificação multiclasse.
        Também pode ser gerado se os dados da classificação binária
        forem inválidos.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("y_true deve ser unidimensional.")

    if y_score.ndim == 1:
        if class_k is not None:
            raise ValueError("class_k deve ser None para classificação binária.")

        return binary_roc_auc(y_true, y_score)

    if y_score.ndim != 2:
        raise ValueError("y_score deve ser unidimensional ou bidimensional.")

    if y_true.size != y_score.shape[0]:
        raise ValueError("O tamanho de y_score deve ser o número de linhas de y_true.")

    if y_score.shape[1] < 2:
        raise ValueError("y_score deve conter ao menos 2 colunas de classes.")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Os valores de y_score devem ser finitos.")

    if class_k is None:
        raise ValueError("class_k é exigida para classificações multiclasse.")

    if isinstance(class_k, (bool, np.bool_)) or not isinstance(
        class_k, (int, np.integer)
    ):
        raise ValueError("class_k deve ser um inteiro.")

    n_classes = y_score.shape[1]

    if class_k < 0 or class_k >= n_classes:
        raise ValueError("class_k está fora do intervalo válido.")

    if not np.all(np.isin(y_true, np.arange(n_classes))):
        raise ValueError(
            "Os labels de y_true devem corresponder aos índices das colunas de y_score."
        )

    y_true_bin = (y_true == class_k).astype(int)
    y_score_bin = y_score[:, class_k]

    return binary_roc_auc(y_true_bin, y_score_bin)
