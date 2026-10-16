import pytest 
import statstech.metrics.classification as mt

def test_confusion_matrix():
    Y_TRUE = [3, 2, 4, 3, 3]
    Y_PRED = [3, 1, 4, 2, 3]
    CLASSES = [1, 2, 3, 4]

    expected = [[0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 0, 0, 1]]

    
    assert mt.confusion_matrix(Y_TRUE, Y_PRED, CLASSES) == expected

