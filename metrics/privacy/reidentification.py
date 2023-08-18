from itertools import combinations  # Standard library
from typing import Tuple, List, Type

# 3rd party packages
import gower
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local
from .base import PrivacyMetric
import utils.draw as udraw
import utils.stats as ustats
import utils.learning as ulearning


def get_metrics() -> List[Type[PrivacyMetric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of privacy metrics
    """

    return [
        DistanceToClosestRecord,
    ]


class DistanceToClosestRecord(PrivacyMetric):
    """
    Check that synthetic data are not copies of real data by ensuring that the Distance to Closest Record (DCR)
    and the Nearest Neighbour Distance Ratio (NNDR) are high.

    See `Zhao, Z., Kunar, A., Birke, R., & Chen, L. Y. (2021, November). Ctab-gan: Effective table data synthesizing.
    In Asian Conference on Machine Learning (pp. 97-112). PMLR."
    JMIR medical informatics 10.4 (2022): e35734 <https://proceedings.mlr.press/v157/zhao21a>`_
    for more details.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real and synthetic datasets
        for better computing performance
    """

    name = "DCR"
    alias = "dcr"

    def __init__(
        self,
        random_state: int = None,
        sampling_frac: float = 0.2,
    ):
        super().__init__(random_state)
        self._sampling_frac = sampling_frac

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "name": "dcr_5th_percent_synthreal",
                "min": 0,
                "max": np.inf,
                "objective": "max",
            },
            {
                "name": "dcr_5th_percent_real",
                "min": 0,
                "max": np.inf,
                "objective": "max",
            },
            {
                "name": "dcr_5th_percent_synth",
                "min": 0,
                "max": np.inf,
                "objective": "max",
            },
            {
                "name": "nndr_5th_percent_synthreal",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "name": "nndr_5th_percent_real",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "name": "nndr_5th_percent_synth",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
        ]
        return submetrics

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the Distance to Closest Record (DCR) between any synthetic sample
        and its closest corresponding real sample and the Nearest Neighbour Distance Ratio (NNDR).

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the 5th percentile DCR and NNDR between synthetic and real
                **dcr(nndr)_5th_percent_synthreal** , as well as within real **dcr(nndr)_5th_percent_real** and
                synthetic **dcr(nndr)_5th_percent_synth** datasets
            * **detailed** -- the DCR and NNDR for each synthetic sample to the closest real sample
                **dcr(nndr)_synthreal**, within real **dcr(nndr)_real** and synthetic **dcr(nndr)_synth** samples
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Sample a fraction of the datasets for computation performance
        real = df_real["test"].sample(frac=self._sampling_frac, replace=False)
        synth = df_synthetic["test"].sample(frac=self._sampling_frac, replace=False)

        # Compute the gower distance (adapted to mixed data)
        cat_features = [  # boolean array instead of column names
            True if col in metadata["categorical"] else False for col in real.columns
        ]
        #   Convert numerical columns to float (otherwise error in the numpy divide)
        real[metadata["continuous"]] = real[metadata["continuous"]].astype("float")
        synth[metadata["continuous"]] = synth[metadata["continuous"]].astype("float")

        pairwise_gower_synthreal = gower.gower_matrix(
            synth, real, cat_features=cat_features
        )
        pairwise_gower_real = gower.gower_matrix(real, cat_features=cat_features)
        pairwise_gower_synth = gower.gower_matrix(synth, cat_features=cat_features)

        # Keep only the 2 smallest distances (first column is 0 for the within real/synth)
        dist_synthreal = np.sort(pairwise_gower_synthreal, axis=1)[:, 0:2]
        dist_real = np.sort(pairwise_gower_real, axis=1)[:, 1:3]
        dist_synth = np.sort(pairwise_gower_synth, axis=1)[:, 1:3]

        # Divide the smallest by the second smallest for NNDR
        ratio_synthreal = np.divide(
            dist_synthreal[:, 0],
            dist_synthreal[:, 1],
            out=np.zeros_like(dist_synthreal[:, 0]),
            where=dist_synthreal[:, 1] != 0,
        )
        ratio_real = np.divide(
            dist_real[:, 0],
            dist_real[:, 1],
            out=np.zeros_like(dist_real[:, 0]),
            where=dist_real[:, 1] != 0,
        )
        ratio_synth = np.divide(
            dist_synth[:, 0],
            dist_synth[:, 1],
            out=np.zeros_like(dist_synth[:, 0]),
            where=dist_synth[:, 1] != 0,
        )

        # Compute the 5th percentile for the average results
        dcr_percent_synthreal = np.percentile(dist_synthreal[:, 0], q=5)
        dcr_percent_real = np.percentile(dist_real[:, 0], q=5)
        dcr_percent_synth = np.percentile(dist_synth[:, 0], q=5)
        nndr_percent_synthreal = np.percentile(ratio_synthreal, q=5)
        nndr_percent_real = np.percentile(ratio_real, q=5)
        nndr_percent_synth = np.percentile(ratio_synth, q=5)

        res = {
            "average": {
                "dcr_5th_percent_synthreal": dcr_percent_synthreal,
                "dcr_5th_percent_real": dcr_percent_real,
                "dcr_5th_percent_synth": dcr_percent_synth,
                "nndr_5th_percent_synthreal": nndr_percent_synthreal,
                "nndr_5th_percent_real": nndr_percent_real,
                "nndr_5th_percent_synth": nndr_percent_synth,
            },
            "detailed": {
                "dcr_synthreal": dist_synthreal[:, 0],
                "dcr_real": dist_real[:, 0],
                "dcr_synth": dist_synth[:, 0],
                "nndr_synthreal": ratio_synthreal,
                "nndr_real": ratio_real,
                "nndr_synth": ratio_synth,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a histogram for DCR and NNDR submetrics.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "dcr_synthreal",
                "dcr_real",
                "dcr_synth",
                "nndr_synthreal",
                "nndr_real",
                "nndr_synth",
            ]
        )

        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=figsize,
            layout="constrained",
            sharex="row",
            sharey="row",
        )

        submetric = ["dcr", "nndr"]
        titles = [
            "Gower Distance to Closest Record",
            "Nearest Neighbour Gower Distance Ratio",
        ]

        for i in range(2):
            udraw.histplot_plot(
                s=pd.Series(report[f"{submetric[i]}_synthreal"]),
                title="",
                value_name=titles[i],
                xrotation=False,
                ax=axes[i][0],
            )
            axes[i][0].legend(["Synthetic to real"])

            udraw.histplot_hue(
                s=pd.Series(report[f"{submetric[i]}_real"]),
                s_nested=pd.Series(report[f"{submetric[i]}_synth"]),
                original_name="Real",
                nested_name="Synthetic",
                hue_name="Within",
                title="",
                value_name=titles[i],
                xrotation=False,
                ax=axes[i][1],
            )
