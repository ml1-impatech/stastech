import pytest
import numpy as np

from statstech.metrics.classification import (
    confusion_matrix,
    true_positives,
    false_positives,
    true_negatives,
    false_negatives,
    accuracy_score,
    precision_score,
    recall_score,
    specificity_score,
    f1_score,
    macro_metric,
    macro_f1,
    macro_specificity,
    macro_recall,
    macro_precision,
    log_loss,
    binary_roc_auc,
    roc_auc_score,
)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([1, 2, 3], [3, 2, 1], [0, 1, 2, 3]),
        ([1, 2, 1], [3, 3, 3], [0, 1, 2, 3]),
    ],
)


def test_confusion_matrix_returns_2d_square_matrix(y_true, y_pred, classes):
    
    c_matrix = confusion_matrix(y_true, y_pred, classes)

    contador = 0
    for i in range(len(c_matrix)):
        contador += 1
        assert len(c_matrix) == len(c_matrix[i])

    c_matrix = np.asarray(c_matrix)
    
    assert c_matrix.shape == (contador, contador)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([0, 1, 1], [1, 0, 0], [0, 1]),
    ],
)


def test_confusion_matrix_counts_correctly(y_true, y_pred, classes):
    c_matrix = confusion_matrix(y_true, y_pred, classes)

    assert c_matrix[0][0] == 0
    assert c_matrix[1][1] == 0
    assert c_matrix[0][1] == 1
    assert c_matrix[1][0] == 2

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([0, 1, 1], [1, 0, 0, 1], [0, 1]),
    ],
)

def test_confusion_matrix_raises_error_for_different_lengths(y_true, y_pred, classes):
        with pytest.raises(ValueError):
            confusion_matrix(y_true, y_pred, classes)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([], [], [0, 1]),
    ],
)

def test_confusion_matrix_raises_error_for_empty_inputs(y_true, y_pred, classes):
             with pytest.raises(ValueError):
                confusion_matrix(y_true, y_pred, classes)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([1, 2, 3], [3, 2, 1], []),
        ([1, 2, 1], [3, 3, 3], []),
    ],
)

def test_confusion_matrix_raises_error_for_empty_classes(y_true, y_pred, classes):
            with pytest.raises(ValueError):
                confusion_matrix(y_true, y_pred, classes)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([1, 2, 3], [3, 2, 1], [0, 0, 1, 2, 3]),
        ([1, 2, 1], [3, 3, 3], [0, 1, 1, 2, 3]),
    ],
)

def test_confusion_matrix_raises_error_for_duplicate_classes(y_true, y_pred, classes):
            with pytest.raises(ValueError):
                confusion_matrix(y_true, y_pred, classes)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([1, 2, 3], [3, 2, 1], [1, 0, 3, 2]),
    ],
)

def test_confusion_matrix_respects_classes_order(y_true, y_pred, classes):
       c_matrix = confusion_matrix(y_true, y_pred, classes)

       assert c_matrix[3][3] == 1 

@pytest.mark.parametrize(
    ("y_true", "y_pred", "classes"),
    [
        ([1, 2, 67], [3, 2, 1], [1, 2, 3]),
        ([1, 2, 1], [3, 3, 3], ['preto', 'branco']),
    ],
)

def test_confusion_matrix_raises_error_for_unknown_label(y_true, y_pred, classes):
            with pytest.raises(KeyError):
                confusion_matrix(y_true, y_pred, classes)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 2),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 0),
    ],
)

def test_true_positives_returns_diagonal_value(c_matrix, class_k):
       tp = true_positives(c_matrix, class_k)
       diagonal = c_matrix[class_k][class_k]
       assert tp == diagonal

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 27),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 5),
    ],
)


def test_true_positives_raises_error_for_invalid_class_index(c_matrix, class_k):
            with pytest.raises(ValueError):
                true_positives(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2], [3, 14, 3]], 1),
        ([[65, 2, 67], [33, 22, 10]], 0),
    ],
)

def test_true_positives_raises_error_for_non_square_matrix(c_matrix, class_k):
            with pytest.raises(ValueError):
                true_positives(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 2.3),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 1.1),
    ],
)

def test_true_positives_raises_error_for_non_integer_class_index(c_matrix, class_k):
            with pytest.raises(ValueError):
                true_positives(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([], 0),
    ],
)

def test_true_positives_raises_error_for_empty_matrix(c_matrix, class_k):
            with pytest.raises(ValueError):
                true_positives(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], False),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], True),
    ],
)


def test_true_positives_raises_error_for_boolean_class_index(c_matrix, class_k):
            with pytest.raises(ValueError):
                true_positives(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 2),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 0),
    ],
)

def test_false_positives_sums_column_except_diagonal(c_matrix, class_k):
       fp = false_positives(c_matrix, class_k)
       column = [c_matrix[i][class_k] for i in range(len(c_matrix))]
       column_minus_class_k = sum(column) - c_matrix[class_k][class_k]
       assert fp == column_minus_class_k

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 2),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 0),
    ],
)

def test_true_negatives_sums_values_outside_class_row_and_column(c_matrix, class_k):
       tn = true_negatives(c_matrix, class_k)
       column = [c_matrix[i][class_k] for i in range(len(c_matrix))]
       row = [c_matrix[class_k][i] for i in range(len(c_matrix))]
       matrix_minus_row_and_column = (sum([sum(c_matrix[i]) for i in range(len(c_matrix))]) 
       - sum(row) - sum(column) + c_matrix[class_k][class_k])
       assert tn == matrix_minus_row_and_column

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 2),
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], 0),
    ],
)

def test_false_negatives_sums_row_except_diagonal(c_matrix, class_k):
       fn = false_negatives(c_matrix, class_k)
       row = [c_matrix[class_k][i] for i in range(len(c_matrix))]
       row_minus_class_k = sum(row) - c_matrix[class_k][class_k]
       assert fn == row_minus_class_k

@pytest.mark.parametrize(
    ("c_matrix"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]]),
    ],
)


def test_accuracy_score_returns_correct_ratio(c_matrix):
       assert accuracy_score(c_matrix) == 0.0625 

@pytest.mark.parametrize(
    ("c_matrix"),
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    ],
)

def test_accuracy_score_raises_error_for_zero_matrix(c_matrix):
        with pytest.raises(ValueError):
            accuracy_score(c_matrix)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 2, 67], [3, 2, 1], [3, 14, 3]], 1),
    ],
)


def test_precision_score_returns_correct_ratio(c_matrix, class_k):
       assert precision_score(c_matrix, class_k) == 1/9

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 0, 67], [14, 0, 2], [3, 0, 3]], 1),
    ],
)

def test_precision_score_raises_error_when_no_positive_predictions(c_matrix, class_k):
        with pytest.raises(ValueError):
            precision_score(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 14, 67], [0, 0, 0], [3, 2, 3]], 1),
    ],
)

def test_recall_score_returns_correct_ratio(c_matrix, class_k):
       assert recall_score(c_matrix, class_k) == 0

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 14, 67], [0, 0, 0], [3, 14, 3]], 1),
    ],
)

def test_recall_score_raises_error_when_no_actual_positives(c_matrix, class_k):
        with pytest.raises(ValueError):
            recall_score(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], 1),
    ],
)

def test_specificity_score_returns_correct_ratio(c_matrix, class_k):
       assert specificity_score(c_matrix, class_k) == pytest.approx(37/51)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[0, 0, 0], [3, 14, 3], [0, 0, 0]], 1),
    ],
)

def test_specificity_score_raises_error_when_no_actual_negatives(c_matrix, class_k):
        with pytest.raises(ValueError):
            specificity_score(c_matrix, class_k)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], 1),
    ],
)

def test_f1_score_returns_correct_ratio(c_matrix, class_k):
        assert f1_score(c_matrix, class_k) == pytest.approx(22/89)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 0, 67], [0, 0, 0], [3, 0, 3]], 1),
    ],
)

def test_f1_score_raises_error_when_denominator_is_zero(c_matrix, class_k):
        with pytest.raises(ValueError):
            f1_score(c_matrix, class_k)

