# Standard library
from typing import List, Tuple, Union
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local packages
from ..base import Report
from . import univariate as uni
from . import bivariate as biv
from . import population as pop
from . import application as app


class UtilityReport(Report):
    """
    Create a report of the utility metrics.

    :cvar metrics_mapping: the dictionary associating the name of the metric to its class
    :vartype metrics_mapping: dict

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param report_filepath: the path of a computed report if available
    :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
    :param cross_learning: the Cross Learning metrics can slow down the report computation.
        Set to *False* to exclude these metrics. Not taken into account if a list of metrics is provided.
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param corr_type: type of correlation to capture for continuous variables ("pearson" or "spearman")
    :param alpha: the significance level for the chi square test
    :param use_gpu: whether to use GPU computation
    """

    metrics_mapping = {
        m.name: m
        for m in uni.get_metrics()
        + biv.get_metrics()
        + app.get_metrics()
        + pop.get_metrics()
    }

    def __init__(
        self,
        dataset_name: str = None,
        df_real: dict[str, pd.DataFrame] = None,
        df_synthetic: dict[str, pd.DataFrame] = None,
        metadata: dict = None,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
        report_filepath: Union[Path, str] = None,
        metrics: List[str] = None,
        cross_learning: bool = False,
        num_repeat: int = 20,
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
        corr_type: str = "pearson",
        alpha: float = 0.05,
        use_gpu: bool = False,
    ):
        super().__init__(
            dataset_name,
            df_real,
            df_synthetic,
            metadata,
            figsize,
            random_state,
            report_filepath,
        )

        # Metrics instantiation with their respective parameters
        params = {
            "random_state": None,
            "corr_type": corr_type,
            "alpha": alpha,
            "num_repeat": num_repeat,
            "num_kfolds": num_kfolds,
            "num_optuna_trials": num_optuna_trials,
            "use_gpu": use_gpu,
        }

        if report_filepath is None:
            self._init_metrics(metrics=metrics, params=params)

            # Remove Cross Learning metrics that can slow down the report computation
            if metrics is None and not cross_learning:
                self._metrics = [
                    metric
                    for metric in self._metrics
                    if not isinstance(metric, pop.CrossLearning)
                ]

        # Personalized size of the figures
        figsize_longer = (figsize[0], figsize[1] * 1.5)
        figsize_larger = (figsize[0] * 1.5, figsize[1])
        self._figsize[uni.ContinuousStatistics.name] = figsize_larger
        self._figsize[uni.CategoricalStatistics.name] = figsize_longer
        self._figsize[biv.PairwiseCorrelationDifference.name] = figsize_larger
        self._figsize[pop.CrossRegression.name] = figsize_larger
        self._figsize[pop.CrossClassification.name] = figsize_larger
