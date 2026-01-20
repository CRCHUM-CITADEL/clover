# Standard library
from typing import Tuple, Callable, Union

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    # mean_squared_error,
    # root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from optuna.trial import Trial

# Backward compatible RMSE
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, **kwargs):
        """RMSE for scikit-learn < 1.4.0"""
        return mean_squared_error(y_true, y_pred, squared=False, **kwargs)


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


def pipeline_prediction(
    trial: Trial,
    predictor: type(xgb.XGBModel),
    preprocessing: ColumnTransformer,
    loss_function: str,
    use_gpu: bool = False,
) -> Pipeline:
    """
    Create a XGBoost classifier or regressor sklearn pipeline within an Optuna trial.

    :param trial: the Optuna trial to specify the hyperparameters to tune
    :param predictor: the predictor object
    :param preprocessing: a list of steps to perform before training the model
    :param loss_function: the loss function of the predictor
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    :return: a Sklearn pipeline
    """

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "catboost",
                predictor(
                    n_estimators=100,
                    eta=trial.suggest_float("eta", 0.001, 0.1, log=True),
                    max_depth=trial.suggest_int("max_depth", 4, 10),
                    subsample=trial.suggest_float("subsample", 0.5, 1),
                    colsample_bytree=trial.suggest_float("colsample_bylevel", 0.5, 1),
                    # tree_method="auto" if not use_gpu else "gpu_hist",
                    # The lines below are relevant when packages will be updated (gpu_hist is deprecated)
                    tree_method="auto" if not use_gpu else "hist",
                    device="cpu" if not use_gpu else "cuda",
                    objective=loss_function,
                    seed=np.random.randint(1000),
                    # verbosity=0,
                    verbosity=1,
                ),
            ),
        ]
    )
    return pipe


def objective_cross_val(
    trial: Trial,
    pipeline: Callable[[Trial], Pipeline],
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    num_kfolds: int,
    scoring: str,
) -> float:
    """
    Run a k-fold cross validation within an Optuna trial.

    :param trial: the Optuna trial to specify the hyperparameters to tune
    :param pipeline: a sequence of the data transformations to apply with a final estimator
    :param df_train: the training input as a Pandas dataframe
    :param y_train: the training ground truth
    :param num_kfolds: the number of folds to tune the hyperparameters of the predictor
    :param scoring: the scoring metric to evaluate the predictor performance on the validation set
    :return: the average cross validation score across the k-folds
    """

    cv_scores = cross_val_score(
        pipeline(trial),
        df_train,
        y_train,
        cv=num_kfolds,
        scoring=scoring,
    )
    score = np.mean(cv_scores)

    return score


def refit_auc_score(
    trial: Trial,
    pipeline: Callable[[Trial], Pipeline],
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, Pipeline]:
    """
    Refit the best estimator and compute the score on the test set. For classification task only.

    :param trial: the Optuna best trial
    :param pipeline: a sequence of the data transformations to apply with a final estimator
    :param df_train: the training input as a Pandas dataframe
    :param y_train: the training ground truth
    :param df_test: the test input as a Pandas dataframe
    :param y_test: the test ground truth
    :return: a tuple, the score, the associated predictions and the fitted pipeline
    """

    pipe = pipeline(trial)
    pipe.fit(df_train, y_train)
    y_pred = pipe.predict_proba(df_test)

    if y_pred.shape[1] == 2:  # binary case, y_pred needs to be (num_samples,)
        y_pred = y_pred[:, 1]
        score = roc_auc_score(y_test, y_pred)
    else:
        score = roc_auc_score(
            y_test, y_pred, multi_class="ovo", labels=range(y_pred.shape[1])
        )

    return score, y_pred, pipe


def refit_rmse_score(
    trial: Trial,
    pipeline: Callable[[Trial], Pipeline],
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, Pipeline]:
    """
    Refit the best estimator and compute the score on the test set. For regression task only.

    :param trial: the Optuna best trial
    :param pipeline: a sequence of the data transformations to apply with a final estimator
    :param df_train: the training input as a Pandas dataframe
    :param y_train: the training ground truth
    :param df_test: the test input as a Pandas dataframe
    :param y_test: the test ground truth
    :return: a tuple, the score (root mean square error), the associated predictions and the fitted pipeline
    """

    pipe = pipeline(trial)
    pipe.fit(df_train, y_train)
    y_pred = pipe.predict(df_test)

    # score = mean_squared_error(y_test, y_pred, squared=False) # deprecated
    score = root_mean_squared_error(y_test, y_pred)

    return score, y_pred, pipe


def hinge_loss(score: float, threshold: float) -> float:
    """
    Compute the hinge loss. Return 0 if the score is below the given threshold.

    :param score: the loss score
    :param threshold: the threshold to consider the loss score null if below
    :return: the hinge loss
    """

    return max(0.0, score - threshold)


# def gpu_predict_proba(pipe, X):
#     """
#     Perform GPU-accelerated prediction probabilities for a sklearn pipeline
#     with a preprocessing step and an XGBoost model.
#     This could be used in certain modules (ex, LOGAN attacks) - it needs to be adapted in cross validation beforehand.
#
#     Args:
#         pipe: sklearn Pipeline with steps ['preprocessing', 'catboost'] where 'catboost' is an XGBClassifier.
#         X: pandas DataFrame or array-like, input data.
#
#     Returns:
#         numpy.ndarray: predicted probabilities for class 1 (binary classification).
#     """
#     # Preprocess input (CPU)
#     X_transformed = pipe.named_steps["preprocessing"].transform(X)
#
#     # Convert to dense if sparse and convert to float32
#     if hasattr(X_transformed, "toarray"):
#         X_transformed = X_transformed.toarray()
#     X_transformed = X_transformed.astype(np.float32)
#
#     # Convert to CuPy array (GPU)
#     X_gpu = cp.asarray(X_transformed)
#
#     # Get XGBoost booster
#     booster = pipe.named_steps["catboost"].get_booster()
#
#     # Run inplace_predict on GPU (raw margin output)
#     raw_preds = booster.inplace_predict(X_gpu)
#
#     # Convert raw margin to probability using sigmoid (binary case)
#     probs = 1 / (1 + np.exp(-cp.asnumpy(raw_preds)))
#
#     # Return probability of positive class (class 1)
#     if probs.ndim == 1:
#         return probs  # shape (n_samples,)
#     else:
#         return probs[:, 1]  # shape (n_samples,)
