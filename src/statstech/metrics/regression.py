def confusion_matrix(y_true: list[float], y_pred: list[float], classes: list[int]) -> list[list[int]]:
    """Calculates the Confusion Matrix for a classification problem
    
    The confusion matrix compares the actual classes with the predicted classses.
    Each row represents an actual class, and each column represents a predicted class.
    
    Parameters
    ----------
    y_true: list[float]
        List of actual target values corresponding to the observations in y_pred.
    y_pred: list[float]
        List of predictions based on observations.
    classes: list[int]
        List of possible classes of the classificator.
        
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

    