# Standard library
from typing import Tuple, List, Type

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local
from ..base import Metric
import clover.utils.external.gower.gower_dist as gower
import clover.utils.draw as udraw


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of privacy metrics
    """

    return [
        DistanceToClosestRecord,
    ]


class DistanceToClosestRecord(Metric):
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
        self._random_state = random_state
        self._sampling_frac = sampling_frac

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """

        submetrics = [
            {
                "submetric": "dcr_5th_percent_synthreal_train",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "dcr_5th_percent_synthreal_control",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "dcr_5th_percent_train_test_ref",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "nndr_5th_percent_synthreal_train",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "nndr_5th_percent_synthreal_control",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "nndr_5th_percent_train_test_ref",
                "min": 0,
                "max": 1,
                "objective": "max",
            },
            {
                "submetric": "ratio_match_synthreal_train",
                "min": 0,
                "max": 1,
                "objective": "min",
            },
            {
                "submetric": "ratio_match_synthreal_control",
                "min": 0,
                "max": 1,
                "objective": "min",
            },
            {
                "submetric": "ratio_match_train_test_ref",
                "min": 0,
                "max": 1,
                "objective": "min",
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
                **dcr_5th_percent_synthreal** and **nndr_5th_percent_synthreal**
            * **detailed** -- the DCR and NNDR for each synthetic sample to the closest real sample
                **dcr(nndr)_synthreal**, within real **dcr(nndr)_real** and synthetic **dcr(nndr)_synth** samples
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Sample a fraction of the datasets for computation performance
        real_train = df_real["train"].sample(
            n=int(self._sampling_frac * len(df_real["test"])),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )
        real_control = df_real["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )
        synth = df_synthetic["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )
        if real_train.shape[1] == 0:
            return {}
        synth = synth[real_train.columns]  # gower needs the same order

        # Compute the gower distance (adapted to mixed data)
        cat_features = [  # boolean array instead of column names
            True if col in metadata["categorical"] else False
            for col in real_train.columns
        ]

        #   Convert numerical columns to float (otherwise error in the numpy divide)
        real_train[metadata["continuous"]] = real_train[metadata["continuous"]].astype(
            "float"
        )
        real_control[metadata["continuous"]] = real_control[
            metadata["continuous"]
        ].astype("float")
        synth[metadata["continuous"]] = synth[metadata["continuous"]].astype("float")

        def pipeline(
            data,
            to_compare=None,
        ):
            ind = 1 if to_compare is None else 0

            # Compute Gower distance
            pairwise_gower = gower.gower_matrix(
                data_x=data, data_y=to_compare, cat_features=cat_features
            )

            # Keep only the 2 smallest distances (first column is 0 for the within real/synth)
            dist = np.sort(pairwise_gower, axis=1)[:, ind + 0 : ind + 2]

            # Divide the smallest by the second smallest for NNDR
            ratio = np.divide(
                dist[:, 0],
                dist[:, 1],
                out=np.zeros_like(dist[:, 0]),
                where=dist[:, 1] != 0,
            )

            # Compute the 5th percentile for the average results
            dcr_percent = np.percentile(dist[:, 0], q=5)
            nndr_percent = np.percentile(ratio, q=5)

            return dist, ratio, dcr_percent, nndr_percent

        (
            dist_synthreal_train,
            ratio_synthreal_train,
            dcr_percent_synthreal_train,
            nndr_percent_synthreal_train,
        ) = pipeline(synth, real_train)

        (
            dist_synthreal_control,
            ratio_synthreal_control,
            dcr_percent_synthreal_control,
            nndr_percent_synthreal_control,
        ) = pipeline(synth, real_control)

        (
            dist_train_test_ref,
            ratio_train_test_ref,
            dcr_percent_train_test_ref,
            nndr_percent_train_test_ref,
        ) = pipeline(real_control, real_train)

        dist_real_train, ratio_real_train, _, _ = pipeline(real_train)
        dist_real_control, ratio_real_control, _, _ = pipeline(real_control)
        dist_synth, ratio_synth, _, _ = pipeline(synth)

        # Count the match ratio
        ratio_match_synthreal_train = np.sum(dist_synthreal_train[:, 0] < 0.01) / len(
            dist_synthreal_train
        )
        ratio_match_synthreal_control = np.sum(
            dist_synthreal_control[:, 0] < 0.01
        ) / len(dist_synthreal_control)

        ratio_match_train_test_ref = np.sum(dist_train_test_ref[:, 0] < 0.01) / len(
            dist_train_test_ref
        )

        res = {
            "average": {
                "dcr_5th_percent_synthreal_train": dcr_percent_synthreal_train,
                "nndr_5th_percent_synthreal_train": nndr_percent_synthreal_train,
                "dcr_5th_percent_synthreal_control": dcr_percent_synthreal_control,
                "nndr_5th_percent_synthreal_control": nndr_percent_synthreal_control,
                "dcr_5th_percent_train_test_ref": dcr_percent_train_test_ref,
                "nndr_5th_percent_train_test_ref": nndr_percent_train_test_ref,
                "ratio_match_synthreal_train": ratio_match_synthreal_train,
                "ratio_match_synthreal_control": ratio_match_synthreal_control,
                "ratio_match_train_test_ref": ratio_match_train_test_ref,
            },
            "detailed": {
                "dcr_synthreal_train": dist_synthreal_train[:, 0],
                "dcr_synthreal_control": dist_synthreal_control[:, 0],
                "dcr_train_test_ref": dist_train_test_ref[:, 0],
                "dcr_real_train": dist_real_train[:, 0],
                "dcr_real_control": dist_real_control[:, 0],
                "dcr_synth": dist_synth[:, 0],
                "nndr_synthreal_train": ratio_synthreal_train,
                "nndr_synthreal_control": ratio_synthreal_control,
                "nndr_train_test_ref": ratio_train_test_ref,
                "nndr_real_train": ratio_real_train,
                "nndr_real_control": ratio_real_control,
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
                "dcr_synthreal_train",
                "dcr_synthreal_control",
                "dcr_real_train",
                "dcr_real_control",
                "dcr_synth",
                "nndr_synthreal_train",
                "nndr_synthreal_control",
                "nndr_real_train",
                "nndr_real_control",
                "nndr_synth",
            ]
        )

        def plot_dcr_nndr(name, title, axes, synthetic=False):
            submetric = ["dcr", "nndr"]
            titles = [
                "Gower Distance to Closest Record",
                "Nearest Neighbour Gower Distance Ratio",
            ]

            for i in range(2):
                bounds_values = np.concatenate(
                    (
                        report[f"{submetric[i]}_{name}_train"],
                        report[f"{submetric[i]}_{name}_control"],
                    )
                )
                if synthetic:
                    bounds_values = np.concatenate(
                        (bounds_values, report[f"{submetric[i]}_synth"])
                    )
                mini = np.min(bounds_values)
                maxi = np.max(bounds_values)

                udraw.histplot_plot(
                    s=pd.Series(report[f"{submetric[i]}_{name}_train"]),
                    title="",
                    value_name=titles[i],
                    stat="proportion",
                    bins=10,
                    binrange=(mini, maxi),
                    xrotation=False,
                    ax=axes[i][0],
                )

                udraw.histplot_plot(
                    s=pd.Series(report[f"{submetric[i]}_{name}_control"]),
                    title="",
                    value_name=titles[i],
                    stat="proportion",
                    bins=10,
                    binrange=(mini, maxi),
                    xrotation=False,
                    ax=axes[i][1],
                )

                if synthetic:
                    udraw.histplot_plot(
                        s=pd.Series(report[f"{submetric[i]}_synth"]),
                        title="",
                        value_name="",
                        stat="proportion",
                        bins=10,
                        binrange=(mini, maxi),
                        xrotation=False,
                        ax=axes_real_synth[i][2],
                    )

            axes[0][0].set_title(title + "Train")
            axes[0][1].set_title(title + "Control")
            if synthetic:
                axes_real_synth[0][2].set_title("Synthetic")

        # Synthreal 2x2 plot
        _, axes_synthreal = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=figsize,
            layout="constrained",
            sharex="row",
            sharey="row",
        )
        plt.suptitle("Synthetic to real")
        plot_dcr_nndr(name="synthreal", title="", axes=axes_synthreal)

        # Real (2x2) and synthetic 2x3 plot
        _, axes_real_synth = plt.subplots(
            ncols=3,
            nrows=2,
            figsize=figsize,
            layout="constrained",
            sharex="row",
            sharey="row",
        )
        plt.suptitle("Within real and synthetic datasets")
        plot_dcr_nndr(
            name="real", title="Real - ", axes=axes_real_synth, synthetic=True
        )

        axes_real_synth[0][0].set_xlabel("")
        axes_real_synth[1][0].set_xlabel("")
