# Standard library
from typing import List, Tuple

# 3rd party packages
import pandas as pd

# Local packages
from ..report import Report
from . import reidentification as reid


class PrivacyReport(Report):
    """
    Create a report of the privacy metrics.

    :cvar metrics_mapping: the dictionary associating the name of the metric to its class
    :vartype metrics_mapping: dict

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
    :param sampling_frac: the fraction of data to sample from real and synthetic datasets
        for better computing performance
    """

    metrics_mapping = {m.name: m for m in reid.get_metrics()}

    def __init__(
        self,
        dataset_name: str,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
        metrics: List[str] = None,
        sampling_frac: float = 0.2,
    ):
        super().__init__(
            dataset_name, df_real, df_synthetic, metadata, figsize, random_state
        )

        # Metrics instantiation with their respective parameters
        params = {"random_state": None, "sampling_frac": sampling_frac}
        self._init_metrics(metrics=metrics, params=params)

        # Personalized size of the figures
        self._figsize[reid.DistanceToClosestRecord.name] = (
            figsize[0] * 1.5,
            figsize[1] * 1.5,
        )
