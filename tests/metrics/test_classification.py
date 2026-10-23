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
        ([[65, 2, 67], [33, 22, 10], [13, 14, 3]], -1),
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

def test_precision_score_warns_and_returns_zero_when_no_positive_predictions(c_matrix, class_k):
        with pytest.warns(UserWarning):
            assert precision_score(c_matrix, class_k) == 0.0


@pytest.mark.parametrize(
    ("c_matrix", "class_k", "exp_value"),
    [
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], 1, 11/50),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], 2, 3/20),
    ],
)
def test_recall_score_returns_correct_ratio(c_matrix, class_k, exp_value):
    assert recall_score(c_matrix, class_k) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("c_matrix", "class_k"),
    [
        ([[1, 14, 67], [0, 0, 0], [3, 14, 3]], 1),
    ],
)

def test_recall_score_warns_and_returns_zero_when_no_actual_positives(c_matrix, class_k):
        with pytest.warns(UserWarning):
            assert recall_score(c_matrix, class_k) == 0.0

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

def test_specificity_score_warns_and_returns_zero_when_no_actual_negatives(c_matrix, class_k):
        with pytest.warns(UserWarning):
            specificity_score(c_matrix, class_k) == 0.0

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

def test_f1_score_warns_and_returns_zero_when_denominator_is_zero(c_matrix, class_k):
        with pytest.warns(UserWarning):
            assert f1_score(c_matrix, class_k) == 0.0

@pytest.mark.parametrize(
    ("c_matrix", "metric", "exp_value"),
    [
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], precision_score, 0.1196209588),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], recall_score, 0.1273983740),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], specificity_score, 0.5840378576),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], f1_score, 0.1069280283),
    ],
)

def test_macro_metric_returns_mean_of_valid_metrics(c_matrix, metric, exp_value):
       assert macro_metric(c_matrix, metric) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("c_matrix", "metric", "exp_value"),
    [
        ([[0, 1, 2], [0, 3, 4], [0, 5, 6]], precision_score, 5/18),
        ([[0, 0, 0], [1, 2, 3], [4, 5, 6]], recall_score, 11/45),
        ([[1, 2, 3], [0, 0, 0], [0, 0, 0]], specificity_score, 7/18),
        ([[0, 0, 0], [0, 1, 2], [0, 3, 4]], f1_score, 82/273),
    ],
)

def test_macro_metric_includes_zero_for_undefined_metrics_and_warns(c_matrix, metric, exp_value):       
       with pytest.warns(UserWarning):
        result = macro_metric(c_matrix, metric)       

       assert result == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("c_matrix", "metric"),
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], precision_score),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], recall_score),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], specificity_score),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], f1_score),
    ],
)

def test_macro_metric_returns_zero_when_all_class_metrics_are_undefined(c_matrix, metric):       
       with pytest.warns(UserWarning):
        assert macro_metric(c_matrix, metric) == 0.0       

@pytest.mark.parametrize(
    ("c_matrix", "metric", "exp_value"),
    [
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], precision_score, 0.1196209588),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], recall_score, 0.1273983740),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], specificity_score, 0.5840378576),
        ([[1, 14, 67], [19, 11, 20], [3, 14, 3]], f1_score, 0.1069280283),
    ],
)

def test_metrics_returns_macro_average_by_default(c_matrix, metric, exp_value):
       assert metric(c_matrix) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "exp_value"),
    [
        ([0, 1], [[0.8, 0.2], [0.1, 0.9]], 0.1642520335),
        ([0, 1], [[0.5, 0.5], [0.5, 0.5]], 0.6931471806),
    ],
)

def test_log_loss_returns_expected_mean(y_true, p_matrix, exp_value):
       assert log_loss(y_true, p_matrix) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([0, 1], [[0.8, 0.1, 0.1], [0.1, 0.89, 0.01], [0.3, 0.3, 0.4]]),
        ([0, 1], [[0.5, 0.2 ,0.3], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]]),
    ],
)

def test_log_loss_raises_error_for_mismatched_lengths(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix) 

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([0, 1, 2], [[0.8, 0.1, 0.1], [-0.01, 0.89, 1.01], [0.3, 0.3, 0.4]]),
        ([0, 1, 2], [[0.1, 0.2 , -0.2], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]]),
    ],
)

def test_log_loss_raises_error_for_probabilities_outside_range(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix) 


@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([0, 1, 2], [[0.8, 0.1, 0.1], [0.01, 0.89, 0.01], [0.3, 0.3, 0.4]]),
        ([0, 1, 2], [[0.1, 0.2 , 0.2], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]]),
    ],
)

def test_log_loss_raises_error_when_probabilities_do_not_sum_to_one(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix)

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([0, 1, 2], [[0.8, 0.1, 0.1], [0.11, 0.89, np.nan], [0.3, 0.3, 0.4]]),
        ([0, 1, 2], [[0.8, 0.2 , np.inf], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]]),
    ],
)

def test_log_loss_raises_error_for_non_finite_probabilities(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix) 

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "eps"),
    [
        ([0, 1, 2], [[0.8, 0.1, 0.1], [0.11, 0.79, 0.1], [0.3, 0.3, 0.4]], -0.3),
        ([0, 1, 2], [[0.7, 0.2 , 0.1], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], 1),
        ([0, 1, 2], [[0.7, 0.2 , 0.1], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], np.inf),
        ([0, 1, 2], [[0.7, 0.2 , 0.1], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], np.nan),
    ],
)

def test_log_loss_raises_error_for_invalid_epsilon(y_true, p_matrix, eps):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix, epsilon=eps)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "lab", "exp_value"),
    [
        ([-0.3, -0.3, 0.9], [[0.8, 0.1, 0.1], [0.11, 0.79, 0.1], [0.3, 0.3, 0.4]], [-0.3, 0.3, 0.9], 1.1155697321),
        (['dog', 'cat', 'cat'], [[0.7, 0.2 , 0.1], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], ['cat', 'dog', 'snake'], 1.1688526324),
    ],
)

def test_log_loss_supports_custom_class_labels(y_true, p_matrix, lab, exp_value):
       assert log_loss(y_true, p_matrix, lab) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "lab"),
    [
        ([-0.3, 0.67, 0.9], [[0.8, 0.1, 0.1], [0.11, 0.79, 0.1], [0.3, 0.3, 0.4]], [-0.3, 0.3, 0.9]),
        (['dog', 'human', 'cat'], [[0.7, 0.2 , 0.1], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], ['cat', 'dog', 'snake']),
    ],
)

def test_log_loss_raises_error_for_unknown_class_label(y_true, p_matrix, lab):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix, lab)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "lab"),
    [
        ([-0.3, 0.9, 0.9], [[0.7, 0.1, 0.1, 0.1], [0.11, 0.69, 0.1, 0.1], [0.2, 0.3, 0.4, 0.1]], [-0.3, 0.3, 0.9, 0.9]),
        (['dog', 'cat', 'cat'], [[0.7, 0.1, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.3, 0.4, 0.2, 0.1]], ['cat', 'dog', 'snake', 'snake']),
    ],
)

def test_log_loss_raises_error_for_duplicate_classes(y_true, p_matrix, lab):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix, lab)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "lab"),
    [
        ([-0.3, 0.9, 0.9], [[[0.7, 0.1, 0.1, 0.1], [0.11, 0.69, 0.1, 0.1], [0.2, 0.3, 0.4, 0.1]]], [-0.3, 0.3, 0.9, 0.1]),
        ([['dog', 'cat', 'cat']], [[0.7, 0.1, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.3, 0.4, 0.2, 0.1]], ['cat', 'dog', 'snake', 'minion']),
    ],
)

def test_log_loss_raises_error_for_invalid_dimensions(y_true, p_matrix, lab):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix, lab)

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([-0.3, 0.9, 0.9], [[], [], []]),
    ],
)

def test_log_loss_raises_error_for_empty_probability_matrix(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix)

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([], np.empty((0, 3))),
    ],
)

def test_log_loss_raises_error_for_empty_observations(y_true, p_matrix):
       with pytest.raises(ValueError):
        log_loss(y_true, p_matrix)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "epsilon"),
    [
        ([1, 0], [[1, 0], [0.1, 0.9]], 0.01),
    ],
)

def test_log_loss_clips_zero_probability_with_epsilon(y_true, p_matrix, epsilon):
    result = log_loss(y_true, p_matrix, epsilon=epsilon)
    expected = -(np.log(epsilon) + np.log(0.1)) / 2

    assert result == pytest.approx(expected)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "exp_value"),
    [
        ([0], [[0.7, 0.1, 0.1, 0.1]], 0.3566749439),
    ],
)

def test_log_loss_returns_correct_value_for_single_observation(y_true, p_matrix, exp_value):
      assert log_loss(y_true, p_matrix) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "p_matrix"),
    [
        ([0, 1, 1], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
    ],
)

def test_log_loss_returns_zero_for_perfect_predictions(y_true, p_matrix):
      assert log_loss(y_true, p_matrix) == pytest.approx(0)

@pytest.mark.parametrize(
    ("y_true", "y_score", "exp_value"),
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
        ([0, 1], [0.5, 0.5], 0.5),
        ([0, 1], [0.9, 0.1], 0),
        ([0, 1], [0.1, 0.9], 1),        
    ],
)
def test_binary_roc_auc_returns_expected_area(y_true, y_score, exp_value):
      assert binary_roc_auc(y_true, y_score) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 0, 0], [0.1, 0.4, 0.35, 0.8]),
        ([0, 0], [0.5, 0.5]),
        ([1, 1], [0.5, 0.5]),
    ],
)
def test_binary_roc_auc_raises_error_when_only_one_class_is_present(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 1, 1], [0.4, 0.35, 0.8]),
        ([0], [0.5, 0.5]),
    ],
)
def test_binary_roc_auc_raises_error_for_different_lengths(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 1], [[0.4, 0.35, 0.8]]),
        ([[0, 1]], [0.5, 0.5]),
    ],
)
def test_binary_roc_auc_raises_error_for_non_1d_inputs(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 0.3], [0.4, 0.35, 0.8]),
        ([0, 2], [0.5, 0.5]),
    ],
)
def test_binary_roc_auc_raises_error_for_invalid_labels(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 1], [0.4, 0.35, np.nan]),
        ([0, 1], [np.inf, 0.5]),
    ],
)
def test_binary_roc_auc_raises_error_for_non_finite_scores(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([], []),
    ],
)
def test_binary_roc_auc_raises_error_for_empty_inputs(y_true, y_score):
       with pytest.raises(ValueError):
        binary_roc_auc(y_true, y_score)


@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        ([0, 1], [0.2, 0.8], 1),
        ([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.7], 0),
    ],
)

def test_roc_auc_score_raises_error_when_class_k_is_set_for_binary(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k", "exp_value"),
    [
        (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            1,
            1.0,
        ),
    ],
)

def test_roc_auc_score_returns_expected_auc_for_multiclass(y_true, y_score, class_k, exp_value):
    assert roc_auc_score(y_true, y_score, class_k) == pytest.approx(exp_value)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
        ),
    ],
)

def test_roc_auc_score_raises_error_when_class_k_is_missing_for_multiclass(y_true, y_score):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            5,
        ),
                (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            -1,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_class_k_out_of_range(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            1.3,
        ),
                (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            True,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_non_integer_class_k(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, -1, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            2,
        ),
                (
            [0, 1, 2, 56, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            1,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_unknown_class_labels(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, 1, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, np.inf, 0.7],
            ],
            2,
        ),
                (
            [0, 1, 2, 0, 1, 2],
            [
                [0.8, np.nan, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
            1,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_non_finite_scores(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)


@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0],
            [
                [0.8],

            ],
            0,
        )

    ],
)

def test_roc_auc_score_raises_error_for_less_than_two_classes(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [[0, 1, 2, 1, 1, 2]],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.7],
            ],
            2,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_non_1d_y_true(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, 1, 1, 2],
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
            ],
            2,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_mismatched_sample_count(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score", "class_k"),
    [
        (
            [0, 1, 2, 1, 1, 2],
            [[
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.7],
            ]],
            2,
        ),
    ],
)

def test_roc_auc_score_raises_error_for_invalid_y_score_dimensions(y_true, y_score, class_k):
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, class_k)

@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        (
            [0, 1, 0],
            [0.8, 0.1, 0.1],
        ),
    ],
)

def test_roc_auc_score_returns_expected_auc_for_binary(y_true, y_score):
    assert roc_auc_score(y_true, y_score) == pytest.approx(0.25)

@pytest.mark.parametrize(
    ("y_true", "p_matrix", "classes"),
    [
        ([0, 1], [[0.8, 0.2], [0.1, 0.9]], [0]),
        ([0, 1], [[0.8, 0.2], [0.1, 0.9]], [0, 1, 2]),
    ],
)
def test_log_loss_raises_error_for_mismatched_class_count(y_true, p_matrix, classes):
    with pytest.raises(ValueError):
        log_loss(y_true, p_matrix, classes)