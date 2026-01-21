# Standard library
import logging
from itertools import combinations
from typing import Tuple, List, Type

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local
from ..base import Metric
import clover.utils.draw as udraw
import clover.utils.stats as ustats
import clover.utils.learning as ulearning

logging.basicConfig(level=logging.INFO)


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [PairwiseCorrelationDifference, PairwiseChiSquareDifference]


class PairwiseCorrelationDifference(Metric):
    """
    Check the preservation of the pairwise Pearson/Spearman correlations between continuous variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    :param corr_type: type of correlation to capture ("pearson" or "spearman")
    """

    name = "Pairwise Correlation Difference"
    alias = "pcd"

    def __init__(self, random_state: int = None, corr_type: str = "pearson"):
        super().__init__(random_state)
        self._corr_type = corr_type.lower()

        if self._corr_type not in ("pearson", "spearman"):
            raise ValueError(
                f"Invalid correlation type: {self._corr_type}, only pearson and spearman are accepted"
            )

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "norm",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "The Frobenius norm of the discrepancy between the Pearson/Spearman correlation matrices in the real and synthetic datasets evaluates the preservation of the relationships between continuous variables.",
                "interpretation": "The norm difference should be small for similar distributions.",
            }
        ]

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the Frobenius norm of the difference between the Pearson/Spearman Correlation matrices of each dataset.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the Frobenius **norm** of the difference between the Pearson/Spearman Correlation matrices
            * **detailed** -- the correlation matrices for the real **corr_real**
              and synthetic datasets **corr_synthetic**

            or *empty* if there are less than two variables in the input dataframes
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # PCD is defined for continuous variables only
        df_real_cont = df_real["train"].drop(columns=metadata["categorical"])
        df_synth_cont = df_synthetic["train"].drop(columns=metadata["categorical"])

        if df_real_cont.shape[1] <= 1:
            return {}

        logging.info(f"Type of correlation computed: {self._corr_type}")

        corr_real = df_real_cont.corr(method=self._corr_type)
        corr_synth = df_synth_cont.corr(method=self._corr_type)
        corr_diff = np.abs(corr_real - corr_synth)
        norm = np.linalg.norm(corr_diff)

        res = {
            "average": {"norm": norm},
            "detailed": {"corr_real": corr_real, "corr_synthetic": corr_synth},
        }

        return res

    @classmethod
    def draw(
        cls,
        report: dict,
        figsize: Tuple[float, float] = None,
    ) -> None:
        """
        Draw a heatmap and a scatterplot of the pairwise Pearson/Spearman correlations of continuous variables.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(key in report for key in ["corr_real", "corr_synthetic"])

        fig, axes = plt.subplots(ncols=2, layout="constrained", figsize=figsize)
        fig.suptitle(f"Metric: {cls.name}")

        corr_real = report["corr_real"]
        corr_synth = report["corr_synthetic"]
        lower_mask = np.tril(np.ones(corr_real.shape)).astype(bool)
        upper_mask = np.triu(np.ones(corr_real.shape)).astype(bool)
        corr_gathered = corr_real.mask(
            cond=lower_mask, other=corr_synth.mask(upper_mask)
        )

        udraw.heat_map(
            data=corr_gathered,
            title=f"Pairwise correlations\n(real top right, synthetic bottom left)",
            vmin=-1,
            vmax=1,
            ax=axes[0],
        )

        list_corr_real = corr_real.mask(lower_mask).unstack().dropna()
        llist_corr_synth = (
            corr_synth.mask(lower_mask).unstack().dropna()
        )  # same triangle to keep data order

        udraw.scatter_plot(
            x=list_corr_real,
            y=llist_corr_synth,
            xlabel=f"Pairwise correlations in real dataset",
            ylabel="Pairwise correlations in synthetic dataset",
            xlim=[0, 1],
            ylim=[0, 1],
            ax=axes[1],
        )
        udraw.identity_line(ax=axes[1], c="g")


class PairwiseChiSquareDifference(Metric):
    """
    Check the preservation of the pairwise relationships between categorical variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    :param alpha: the significance level for the chi square test
    """

    name = "Pairwise Chi Square Difference"
    alias = "pcsd"

    def __init__(self, random_state: int = None, alpha: float = 0.05):
        super().__init__(random_state)
        self._alpha = alpha

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "sensitivity",
                "min": 0,
                "max": 1,
                "objective": "max",
                "description": "This metric evaluates the preservation of relationships between categorical variables by comparing the decision of rejecting the null hypothesis (H0: there is no relationship between two variables) between the real and synthetic datasets.",
                "interpretation": "Sensitivity is the percentage of true relationships detected among all real relationships: TP / (TP + FN).",
            },
            {
                "submetric": "specificity",
                "min": 0,
                "max": 1,
                "objective": "max",
                "description": "This metric evaluates the preservation of relationships between categorical variables by comparing the decision of rejecting the null hypothesis (H0: there is no relationship between two variables) between the real and synthetic datasets.",
                "interpretation": "Specificity is the percentage of true non-existing relationships detected among all real non-existing relationships: TN / (TN + FP).",
            },
        ]

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the sensitivity and specificity of the chi square H0 rejection (no relationship between the two
        variables) of the real variables pairwise dependencies (*y_true*) and
        the synthetic variables pairwise dependencies (*y_pred*).

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the **sensitivity** and **specificity** scores
            * **detailed** -- the chi square results for real data **chisquare_real** and
              synthetic data **chisquare_synthetic**

            or *empty* if there are less than two variables in the input dataframes
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # PCSD is defined for categorical variables only
        df_real_cat = df_real["train"].drop(columns=metadata["continuous"])
        df_synth_cat = df_synthetic["train"].drop(columns=metadata["continuous"])

        if df_real_cat.shape[1] <= 1:
            return {}

        # Find all pairwise relations
        pairs = combinations(df_real_cat.columns, 2)

        # Compute chi square statistics for both real and synthetic datasets
        chisquare_real = []
        chisquare_synth = []
        for var1, var2 in pairs:
            chisquare_real.append(
                [
                    var1,
                    var2,
                    *ustats.scipy_chi2_contingency(
                        df_real_cat[var1].to_numpy(), df_real_cat[var2].to_numpy()
                    ),
                ]
            )
            chisquare_synth.append(
                [
                    var1,
                    var2,
                    *ustats.scipy_chi2_contingency(
                        df_synth_cat[var1].to_numpy(), df_synth_cat[var2].to_numpy()
                    ),
                ]
            )
        df_real_chisquare = pd.DataFrame(
            chisquare_real, columns=["var1", "var2", "chi2", "p", "dof", "expected"]
        )
        df_synthetic_chisquare = pd.DataFrame(
            chisquare_synth, columns=["var1", "var2", "chi2", "p", "dof", "expected"]
        )

        # Check H0 hypothesis (no relationship between the two variables) - 1 for rejection
        df_real_chisquare["h0_rejection"] = (
            df_real_chisquare["p"] < self._alpha
        ).astype(int)
        df_synthetic_chisquare["h0_rejection"] = (
            df_synthetic_chisquare["p"] < self._alpha
        ).astype(int)

        # Compute the sensitivity and specificy based on the H0 rejection
        tn, fp, fn, tp, sensitivity, specificity = ulearning.sklearn_confusion_matrix(
            df_real_chisquare["h0_rejection"].to_numpy(),
            df_synthetic_chisquare["h0_rejection"].to_numpy(),
        )

        res = {
            "average": {"sensitivity": sensitivity, "specificity": specificity},
            "detailed": {
                "chisquare_real": df_real_chisquare,
                "chisquare_synthetic": df_synthetic_chisquare,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a heatmap of the chi square pairwise H0 rejection difference between real and synthetic
        for categorical variables.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(key in report for key in ["chisquare_real", "chisquare_synthetic"])

        plt.figure(figsize=figsize, layout="constrained")

        df_diff = (
            pd.merge(  # alignment
                report["chisquare_real"],
                report["chisquare_synthetic"],
                on=["var1", "var2"],
                suffixes=("_real", "_synth"),
            )
            .assign(
                diff_h0r=lambda df: df["h0_rejection_real"] - df["h0_rejection_synth"]
            )
            .filter(items=["var1", "var2", "diff_h0r"])  # formatting
            .pivot(index="var1", columns="var2", values="diff_h0r")
        )

        udraw.heat_map(
            data=df_diff,
            title=f"Metric: {cls.name}\n(H0 rejection real - H0 rejection synthetic)",
            vmin=-1,
            vmax=1,
        )
