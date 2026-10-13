def confusion_matrix(y_true: list[float], y_pred: list[float], classes: list[int]) -> list[list[int]]:
    """Calculate the Confusion Matrix for a classification problem.
    
    The confusion matrix compares the actual classes with the predicted classses.
    Each row represents an actual class, and each column represents a predicted class.
    
    Parameters
    ----------
    y_true: list[float]
        List of actual target values corresponding to the observations in y_pred.
    y_pred: list[float]
        List of predictions based on observations.
    classes: list[int]
        List of possible classes of the classifier.
        
    Returns
    -------
    c_matrix: list[list[int]] 
        A square matrix of shape (n_classes, n_classes), where each element in position 
        [i][j] represents the number of observations whose actual class is i and predicted
        class is j.
    
    Raises
    ------
    ValueError
        If y_true and y_pred have different lenghts, or if either list is empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The lists of input predictions and target values" \
        " must have the same length.")
    if len(y_true) == 0:
        raise ValueError("The input lists must not be empty")
    n_classes = len(classes)
    c_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for i in range(len(y_true)):
        c_matrix[y_true[i]][y_pred[i]] += 1
    return c_matrix

def TP(c_matrix: list[list[int]], class_k: int) -> int:
    """Calculate the True Positives of a confusion matrix for a especific class.

    Parameters
    ----------
    c_matrix: list[list[int]]
        A square matrix of shape (n_classes, n_classes), where each element in position 
        [i][j] represents the number of observations whose actual class is i and predicted
        class is j.
    class_k: int
        The index of the class for which the True Positives will be calculated.

    Returns
    -------
    true_positives: int
        The element c_matrix[class_k][class_k], which indicates how many times the classifier
        returned class_k when it was the real class of the observation. 
    """
    true_positives = c_matrix[class_k][class_k]
    return true_positives

def FP(c_matrix: list[list[int]], class_k: int) -> int:
    """Calculate the False Positives of a confusion matrix for a especific class.

    Parameters
    ----------
    c_matrix: list[list[int]]
        A square matrix of shape (n_classes, n_classes), where each element in position 
        [i][j] represents the number of observations whose actual class is i and predicted
        class is j.
    class_k: int
        The index of the class for which the False Positives will be calculated.

    Returns
    -------
    false_positives: int
        Sum of the entries in column class_k, excluding c_matrix[class_k][class_k].
        This represents the number of observations predicted as class_k whose actual
        class is different from class_k.
    """
    false_positives = 0
    for i in range(len(c_matrix)):
        if i != class_k:
            false_positives += c_matrix[i][class_k]
    return false_positives

def TN(c_matrix: list[list[int]], class_k: int) -> int:
    """Calculate the True Negatives of a confusion matrix for a especific class.
 
    Parameters
    ----------
    c_matrix: list[list[int]]
        A square matrix of shape (n_classes, n_classes), where each element in position 
        [i][j] represents the number of observations whose actual class is i and predicted
        class is j.
    class_k: int
        The index of the class for which the True Negatives will be calculated.

    Returns
    -------
    true_negatives: int
        Sum of all entries outside row class_k and column class_k. This represents the number 
        of observations whose actual and predicted classes are both different from class_k.
    """
    true_negatives = 0
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            if i != class_k and j != class_k:
                true_negatives += c_matrix[i][j]
    return true_negatives

def FN(c_matrix: list[list[int]], class_k: int) -> int:
    """Calculate the False Negatives of a confusion matrix for a especific class.
 
    Parameters
    ----------
    c_matrix: list[list[int]]
        A square matrix of shape (n_classes, n_classes), where each element in position 
        [i][j] represents the number of observations whose actual class is i and predicted
        class is j.
    class_k: int
        The index of the class for which the False Negatives will be calculated.

    Returns
    -------
    false_negatives: int
        Sum of entries in row class_k, excluding c_matrix[class_k][class_k]. This represents
        the number of observations whose actual class is class_k but whose predicted class
        is different from class_k.
    """
    true_negatives = 0
    for j in range(len(c_matrix)):
        if j != class_k:
            true_negatives += c_matrix[class_k][j]
    return true_negatives

