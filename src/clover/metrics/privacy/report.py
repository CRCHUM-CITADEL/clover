# Standard library
from typing import List, Tuple, Union
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local packages
from ..base import Report
from . import reidentification as reid
from . import membership as mem


class PrivacyReport(Report):
    """
    Create a report of the privacy metrics.

    :cvar metrics_mapping: the dictionary associating the name of the metric to its class
    :vartype metrics_mapping: dict

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** (if MIAs are applied) sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param report_filepath: the path of a computed report if available
    :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
    :param sampling_frac: the fraction of data to sample from real and synthetic datasets
        for better computing performance
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    """

    metrics_mapping = {m.name: m for m in reid.get_metrics() + mem.get_metrics()}

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
        sampling_frac: float = 0.2,
        num_repeat: int = 10,
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
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
            "random_state": random_state,
            "sampling_frac": sampling_frac,
            "num_repeat": num_repeat,
            "num_kfolds": num_kfolds,
            "num_optuna_trials": num_optuna_trials,
            "use_gpu": use_gpu,
        }
        if report_filepath is None:
            self._init_metrics(metrics=metrics, params=params)
