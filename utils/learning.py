from typing import Tuple, List, Union  # standard library

import numpy as np  # 3rd party packages
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


def sklearn_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int, float, float]:
    """
    Compute the confusion matrix in a binary setup, the sensitivity and the specificity (see https://scikit-learn.org).

    :param y_true: the binary ground truth
    :param y_pred: the binary predictions
    :return: the true negative, false positive, false negative and true positive counts
        as well as the sensitivity and the specificity
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan

    return tn, fp, fn, tp, sensitivity, specificity


def train_predict(
    pipeline: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_list: List[np.ndarray],
    y_test_list: List[np.ndarray],
    classif_labels: Union[List[int], np.ndarray] = None,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Train a classifier or a regressor and score the predictions for the test sets.

    :param pipeline: a sequence of the data transformations to apply with a final estimator
    :param x_train: the training inputs
    :param y_train: the training ground truth
    :param x_test_list: the test sets for the prediction
    :param y_test_list: the test sets ground truth
    :param classif_labels: list of labels if a classifier is trained,
        must be specified even for a one-class classification
    :return: the predictions scores and the raw predictions
    """
    pipeline.fit(x_train, y_train)

    scores = []
    preds = []
    for x_test, y_test in zip(x_test_list, y_test_list):
        if classif_labels is not None:
            y_pred = pipeline.predict_proba(x_test)
            labels = classif_labels
            if y_pred.shape[1] == 2:  # binary case, y_pred needs to be (num_samples,)
                y_pred = y_pred[:, 1]
                labels = None
            if len(np.unique(y_test)) != 1:  # roc auc score not defined for one class
                score = roc_auc_score(y_test, y_pred, multi_class="ovo", labels=labels)
            else:
                score = 1
        else:  # regression
            y_pred = pipeline.predict(x_test)
            score = mean_squared_error(y_test, y_pred)
        scores.append(score)
        preds.append(y_pred)

    return scores, preds


def hinge_loss(score: float, threshold: float) -> float:
    """
    Compute the hinge loss. Return 0 if the score is below the given threshold.

    :param score: the loss score
    :param threshold: the threshold to consider the loss score null if below
    :return: the hinge loss
    """

    return max(0.0, score - threshold)
