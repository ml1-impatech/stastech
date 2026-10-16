from collections.abc import Callable
import numpy as np
import numpy.typing as npt

def confusion_matrix(y_true: list[int], y_pred: list[int], classes: list[int]) -> list[list[int]]:
    """Calcula a matriz de confusão para um problema de classificação.
    
    A matriz de confusão compara as classes reais com as classes previstas. 
    Cada linha representa uma classe verdadeira, e cada coluna uma classe prevista.
    
    Parameters
    ----------
    y_true: list[float]
        Lista dos valores reais correspondentes às previsões em y_pred.
    y_pred: list[float]
        Lista de predições baseadas nas observações.
    classes: list[int]
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
    """
    if len(y_true) != len(y_pred):
        raise ValueError("A lista de predições e de valores reais devem ter o mesmo comprimento.")
    if len(y_true) == 0:
        raise ValueError("As listas de input devem ser não vazias.")
    n_classes = len(classes)
    dict_classes = {}
    for i in range(n_classes):
        dict_classes[classes[i]] = i
    c_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
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
    """
    true_positives = c_matrix[class_k][class_k]
    return true_positives

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
    """
    false_positives = 0
    for i in range(len(c_matrix)):
        if i != class_k:
            false_positives += c_matrix[i][class_k]
    return false_positives

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
    """
    true_negatives = 0
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            if i != class_k and j != class_k:
                true_negatives += c_matrix[i][j]
    return true_negatives

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
    """
    true_negatives = 0
    for j in range(len(c_matrix)):
        if j != class_k:
            true_negatives += c_matrix[class_k][j]
    return true_negatives

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
    """
    if len(c_matrix) == 0:
        raise ValueError("A matriz de confusão deve ter ao menos um elemento.")
    true_predictions = 0
    total = 0
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            if i == j:
                true_predictions += c_matrix[i][j]
            total += c_matrix[i][j]
    if total == 0:
        raise ValueError("A matriz de confusão deve ter ao menos um elemento diferente de zero.")
    return true_predictions/total


def precision_score(c_matrix: list[list[int]], class_k: int) -> float:
    """Calcula o Precision Score para uma matriz de confusão. 

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j. 
    class_k: int
        O índice da classe para qual o Precision será calculado.

    Returns
    -------
    tp/(tp+fp): float
        A proporção das observações previstas como pertencentes a class_k
        que realmente pertencem a essa classe.
    
    Raises
    ------
    ValueError
        Se tp+fp for igual a zero, tornando a métrica indefinida.
    """
    tp = true_positives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    if tp+fp != 0:
        return tp/(tp+fp)
    else:
        raise ValueError("tp + fp deve ser diferente de 0.")
    
def recall_score(c_matrix: list[list[int]], class_k: int) -> float:
    """Calcula o Recall Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.    
    class_k: int
        O índice da classe para qual o Recall será calculado.

    Returns
    -------
    tp/(tp+fn): float
        A proporção entre True Positive e todas as observações que deveriam ser positivas.
        Isto indica o quão bem um classificador identifica observações pertencentes a class_k.
    
    Raises
    ------
    ValueError
        Se tp+fn for igual a zero, tornando a métrica indefinida.
    """
    tp = true_positives(c_matrix, class_k)
    fn = false_negatives(c_matrix, class_k)
    if tp+fn != 0:
        return tp/(tp+fn)
    else:
        raise ValueError("tp + fn deve ser diferente de 0.")

def specificity_score(c_matrix: list[list[int]], class_k: int) -> float:
    """Calcula o Specificity Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.    
    class_k: int
        O índice da classe para qual o Specificity score será calculado.

    Returns
    -------
    tn/(tn+fp): float
        A razão entre True Negatives e todas as observações que deveriam ser negativas.
        Isto indica o quão bem um classificador identifica observaçõse que não pertencem
        a class_k.

    Raises
    ------
    ValueError
        Se tn+fp for igual a zero, tornando a métrica indefinida.
    """    
    tn = true_negatives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    if tn+fp != 0:
        return tn/(tn+fp)
    else: 
        raise ValueError("tn + fp deve ser diferente de 0.")

def f1_score(c_matrix: list[list[int]], class_k: int) -> float:
    """Calcula o F1 Score para uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.    
    class_k: int
        O índice da classe para qual o f1 score será calculado.

    Returns
    -------
    tp*2/(2*tp+fn+fp): float
        O F1 Score, que é a média harmônica entre o Precision e o Recall. Isto combina
        as duas métricas em um só Score.
    
    Raises
    ------
    ValueError
        Se o denominador do f1 for igual a zero, tornando a métrica indefinida.
    """    
    tp = true_positives(c_matrix, class_k)
    fp = false_positives(c_matrix, class_k)
    fn = false_negatives(c_matrix, class_k)
    if 2*tp + fn + fp != 0:
        return 2*tp/(2*tp + fn + fp)
    else: 
        raise ValueError("O denominador 2*tp + fn + fp deve ser diferente de zero.")

def macro_metric(c_matrix: list[list[int]], metric: Callable[[list[list[int]], int], float]) -> float:
    """Calcula o média macro para uma métrica específica em uma matriz de confusão.

    Parameters
    ----------
    c_matrix: list[list[int]]
        Uma matriz quadrada de dimensões (n_classes, n_classes), na qual cada elemento
        na posição [i][j] representa o número de observações tais que a verdadeira classe
        é i e a classe prevista é j.    
    metric: callable
        A métrica que será utilizada para calcular seu macro.

    Returns
    -------
    t_metric: float
        A média aritmética da métrica calculada para cada classe.
    
    Raises
    ------
    ValueError
        Se existe ValueError para toda classe em c_matrix, causando métrica indefinida
        para toda iteração do "for". Ignora-se classes com valor da métrica indefinido
        ou classes com ValueError por algum outro motivo.
    """    
    t_metric = 0
    n_valid = 0
    n_classes = len(c_matrix)
    for i in range(n_classes):
        try:
            t_metric += metric(c_matrix, i)
            n_valid += 1
        except ValueError:
            continue
    if n_valid == 0:
        raise ValueError("Nenhuma classe possui métrica definida.")
    t_metric /= n_valid 
    return t_metric

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
        Ignora-se classes com valor f1 indefinido.
    
    Raises
    ------
    ValueError
        Se existe ValueError para toda classe em c_matrix, causando métrica indefinida
        para toda iteração do "for".
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
        Ignora-se classes com valor indefinido.
    
    Raises
    ------
    ValueError
        Se existe ValueError para toda classe em c_matrix, causando métrica indefinida
        para toda iteração do "for".
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
        Ignora-se classes com valor indefinido.
    
    Raises
    ------
    ValueError
        Se existe ValueError para toda classe em c_matrix, causando métrica indefinida
        para toda iteração do "for".
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
        Ignora-se classes com valor indefinido.
    
    Raises
    ------
    ValueError
        Se existe ValueError para toda classe em c_matrix, causando métrica indefinida
        para toda iteração do "for".
    """
    return macro_metric(c_matrix, precision_score)


def log_loss(y_true: npt.ArrayLike, p_matrix: npt.ArrayLike, epsilon: float=1e-10) -> float:
    """Calcula a perda logarítmica de um classificador.

    Parameters
    ----------
    y_true: array-like
        Array com os índices das classes reais do alvo, entre 0 e n_classes - 1.
    p_matrix: array-like
        Uma matriz de dimensões (n_samples, n_classes), na qual cada elemento
        na posição [i][j] representa a probabilidade de a observação i ser
        prevista como classe j.
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
        Se os índices das classes forem inválidos, se houver probabilidades
        inválidas ou se o valor de epsilon for inválido.
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
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon deve ser finito e estar entre 0 e 1.")
    if len(y_true) != n_samples:
        raise ValueError("As dimensões de p_matrix e y_true devem ser compatíveis.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true deve conter índices inteiros de classes.")
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError("y_true contém índices fora do intervalo.")
    if not np.all(np.isfinite(p_matrix)):
        raise ValueError("p_matrix não pode conter NaN ou infinito.")
    if np.any(p_matrix < 0) or np.any(p_matrix > 1):
        raise ValueError("p_matrix contém probabilidades fora do intervalo.")
    if not np.allclose(p_matrix.sum(axis=1), 1.0):
        raise ValueError("As probabilidades de cada linha devem somar 1.")

    total = 0.0

    for i in range(len(y_true)):
        ind = y_true[i]
        prob = np.clip(p_matrix[i, ind], epsilon, 1.0)
        total += np.log(prob)

    l_loss = -total / len(y_true)
    return float(l_loss)

def roc_auc_score(y_true, y_score):
    epsilon = 1e-10
    threshholds = []
    for i in range(len(y_score)):
        threshholds.append(y_score[i])
        