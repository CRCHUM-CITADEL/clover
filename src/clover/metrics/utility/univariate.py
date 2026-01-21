# Standard library
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Type

# 3rd party packages
import pandas as pd
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# Local
from ..base import Metric
import clover.utils.stats as ustats
import clover.utils.draw as udraw
import clover.utils.preprocessing as upreprocessing


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [
        ContinuousConsistency,
        CategoricalConsistency,
        ContinuousStatistics,
        CategoricalStatistics,
        ContinuousUnivariateHellingerDistance,
        CategoricalUnivariateHellingerDistance,
        ContinuousUnivariateKLDivergence,
        CategoricalUnivariateKLDivergence,
    ]


class Consistency(Metric, metaclass=ABCMeta):
    """
    Check that the synthetic data are within values of the real data.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "Consistency"
    alias = "consis"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max**, **objective**, **description** and **interpretation**)
        """
        return [
            {
                "submetric": "within_ratio",
                "min": 0,
                "max": 1,
                "objective": "max",
                "description": "The consistency metric evaluates whether synthetic data values fall within the bounds of the real data for continuous variables and whether categories match those in the real data for categorical variables.",
                "interpretation": "A ratio of 1 indicates full adherence.",
            }
        ]

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot of the ratios.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert "within_ratio" in report

        info = cls.get_average_submetrics()[0]
        assert info["submetric"] == "within_ratio"

        plt.figure(figsize=figsize, layout="constrained")

        udraw.bar_plot(
            data=pd.DataFrame([report["within_ratio"]]),
            title=f"Metric: {cls.name}\n",
            value_name="Within the real data ratio",
            orient="v",
            lim=(info["min"], info["max"]),
        )


class ContinuousConsistency(Consistency):
    """
    Check that the synthetic data are within the bounds of the real data.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "Continuous " + Consistency.name
    alias = "cont_" + Consistency.alias

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the number of synthetic data samples between the minimum and maximum of the real data.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average ratio **within_ratio** of the number of synthetic data samples
              between the minimum and maximum of the real data across all continuous variables
            * **detailed** -- a dictionary **within_ratio** with the variables names as keys
              and the ratios as values
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for continuous variables only
        df_real_cont = df_real["train"].drop(columns=metadata["categorical"])
        df_synth_cont = df_synthetic["train"].drop(columns=metadata["categorical"])

        if df_real_cont.shape[1] == 0:
            return {}

        between_minmax_ratio = {}
        for col in df_real_cont.columns:
            min_real = df_real_cont[col].min()
            max_real = df_real_cont[col].max()

            in_between = np.sum(
                (df_synth_cont[col] >= min_real) & (df_synth_cont[col] <= max_real)
            )
            between_minmax_ratio[col] = in_between / len(df_real_cont[col])

        res = {
            "average": {
                "within_ratio": np.mean(list(between_minmax_ratio.values())),
            },
            "detailed": {
                "within_ratio": between_minmax_ratio,
            },
        }

        return res


class CategoricalConsistency(Consistency):
    """
    Check that the synthetic data are within the categories of the real data.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "Categorical " + Consistency.name
    alias = "cat_" + Consistency.alias

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the number of synthetic data samples within the categories of the real data.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average ratio **within_ratio**, measuring the coverage of synthetic data samples
              across the categories observed in the real data, averaged over all categorical variables
            * **detailed** -- a dictionary **within_ratio** with the variables names as keys
              and the ratios as values
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for categorical variables only
        df_real_cat = df_real["train"].drop(columns=metadata["continuous"])
        df_synth_cat = df_synthetic["train"].drop(columns=metadata["continuous"])

        if df_real_cat.shape[1] == 0:
            return {}

        within_ratio = {}
        for col in df_real_cat.columns:
            real_table = ustats.frequency_table(df_real_cat[col], "real")
            synth_table = ustats.frequency_table(df_synth_cat[col], "synth")

            df_counts = pd.merge(
                real_table,
                synth_table,
                how="left",  # keep only real data categories
                left_index=True,
                right_index=True,
            ).fillna(0)

            within_ratio[col] = df_counts["synth_freq"].sum()

        res = {
            "average": {
                "within_ratio": np.mean(list(within_ratio.values())),
            },
            "detailed": {
                "within_ratio": within_ratio,
            },
        }

        return res


class ContinuousStatistics(Metric):
    """
    Check that the synthetic data match the median and IQR statistics for continuous variables of the real data.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "Continuous Statistics"
    alias = "cont_stats"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "median_l1_distance",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "The median L1 distance evaluates the average L1 distance between the medians of real and synthetic data across all continuous variables.",
                "interpretation": "It can be compared to the difference in medians between the train and test sets. Note that the variables are not scaled.",
            },
            {
                "submetric": "median_l1_distance_train_test_ref",
                "min": 0,
                "max": np.inf,
                "objective": "min",
            },
            {
                "submetric": "iqr_l1_distance",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "The IQR L1 distance evaluates the average L1 distance between the IQRs of real and synthetic data across all continuous variables.",
                "interpretation": "It can be compared to the difference in IQRs between the train and test sets. Note that the variables are not scaled.",
            },
            {
                "submetric": "iqr_l1_distance_train_test_ref",
                "min": 0,
                "max": np.inf,
                "objective": "min",
            },
        ]

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compare the median and IQR of the synthetic versus real data by computing the *L1* distance.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average *L1* distances **median_l1_distance** and **iqr_l1_distance**
              between the medians and the IQRs respectively of real and synthetic data across all continuous variables
            * **detailed** -- the real **df_real** and synthetic **df_synthetic** raw dataframes
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for continuous variables only
        df_real_cont = df_real["train"].drop(columns=metadata["categorical"])
        df_synth_cont = df_synthetic["train"].drop(columns=metadata["categorical"])

        df_real_cont_test = df_real["test"].drop(columns=metadata["categorical"])

        if df_real_cont.shape[1] == 0:
            return {}

        median_l1_distance = {}
        iqr_l1_distance = {}

        median_l1_distance_train_test_ref = {}
        iqr_l1_distance_train_test_ref = {}

        for col in df_real_cont.columns:
            min_real = df_real_cont[col].min()
            max_real = df_real_cont[col].max()
            mini = min(min_real, df_synth_cont[col].min())
            maxi = max(max_real, df_synth_cont[col].max())
            norm_range = maxi - mini
            if norm_range == 0:  # divide by 0
                norm_range = 1

            median_l1_distance[col] = (
                abs(df_synth_cont[col].median() - df_real_cont[col].median())
                / norm_range
            )

            q75_real, q25_real = np.percentile(df_real_cont[col], [75, 25])
            q75_synth, q25_synth = np.percentile(df_synth_cont[col], [75, 25])
            iqr_l1_distance[col] = (
                abs((q75_synth - q25_synth) - (q75_real - q25_real)) / norm_range
            )

            # Run the same measurement for the reference value (between real train and test)

            mini_train_test_ref = min(min_real, df_real_cont_test[col].min())
            maxi_train_test_ref = max(max_real, df_real_cont_test[col].max())
            norm_range_train_test_ref = maxi_train_test_ref - mini_train_test_ref
            if norm_range_train_test_ref == 0:  # divide by 0
                norm_range_train_test_ref = 1

            median_l1_distance_train_test_ref[col] = (
                abs(df_real_cont_test[col].median() - df_real_cont[col].median())
                / norm_range_train_test_ref
            )

            q75_real_train_test_ref, q25_real_train_test_ref = np.percentile(
                df_real_cont[col], [75, 25]
            )
            q75_synth_train_test_ref, q25_synth_train_test_ref = np.percentile(
                df_real_cont_test[col], [75, 25]
            )
            iqr_l1_distance_train_test_ref[col] = (
                abs(
                    (q75_synth_train_test_ref - q25_synth_train_test_ref)
                    - (q75_real_train_test_ref - q25_real_train_test_ref)
                )
                / norm_range_train_test_ref
            )

        res = {
            "average": {
                "median_l1_distance": np.mean(list(median_l1_distance.values())),
                "median_l1_distance_train_test_ref": np.mean(
                    list(median_l1_distance_train_test_ref.values())
                ),
                "iqr_l1_distance": np.mean(list(iqr_l1_distance.values())),
                "iqr_l1_distance_train_test_ref": np.mean(
                    list(iqr_l1_distance_train_test_ref.values())
                ),
            },
            "detailed": {
                "df_real": df_real_cont,
                "df_synthetic": df_synth_cont,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a boxplot for each continuous variables of both real and synthetic datasets.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(key in report for key in ["df_real", "df_synthetic"])

        # One plot per variable
        num_cols = report["df_real"].shape[1]
        max_plot_per_window = 7
        num_win = num_cols // max_plot_per_window
        if num_cols % max_plot_per_window != 0:
            num_win += 1

        axes = []
        for f in range(num_win):
            fig, axes_f = plt.subplots(
                ncols=max_plot_per_window, figsize=figsize, layout="constrained"
            )
            fig.suptitle(f"Metric: {cls.name} ({f + 1}/{num_win})")
            axes.extend(axes_f)

        for i, col in enumerate(report["df_real"].columns):
            udraw.box_plot_per_column_hue(
                df=report["df_real"][col],
                df_nested=report["df_synthetic"][col],
                original_name="Real",
                nested_name="Synthetic",
                hue_name="Data",
                orient="v",
                title=col,
                value_name="Value",
                ax=axes[i],
            )
            if i % max_plot_per_window != 0:
                axes[i].get_legend().remove()


class CategoricalStatistics(Metric):
    """
    Check that the synthetic data match the support and frequency coverage for categorical variables of the real data.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "Categorical Statistics"
    alias = "cat_stats"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "support_coverage",
                "min": 0,
                "max": 1,
                "objective": "max",
                "description": "The support coverage evaluates the mean of the proportion of represented (non-zero counts) categories in the synthetic data for all categorical variables.",
                "interpretation": "A support coverage smaller than 1 indicates missing categories.",
            },
            {
                "submetric": "frequency_coverage",
                "min": 0,
                "max": 1,
                "objective": "max",
                "description": "The frequency coverage evaluates whether categories appear with the same frequency in the synthetic and real data. It is equal to 1 - the mean of the differences in frequency for all categorical variables.",
                "interpretation": "A high coverage indicates a close representation of categorical variables in the synthetic data.",
            },
        ]

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Measure the coverage of each categorical variable of the synthetic data with reference to the real data.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average **support_coverage** and **frequency_coverage** of the synthetic data
              across all categorical variables
            * **detailed** -- two dictionaries **real_counts** and **synthetic_counts** with the categorical
              variables names as keys and the frequency table of real and synthetic data as values respectively
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for categorical variables only
        df_real_cat = df_real["train"].drop(columns=metadata["continuous"])
        df_synth_cat = df_synthetic["train"].drop(columns=metadata["continuous"])

        if df_real_cat.shape[1] == 0:
            return {}

        support_coverage = {}
        frequency_coverage = {}
        real_counts = {}
        synth_counts = {}

        for col in df_real_cat.columns:
            real_table = ustats.frequency_table(df_real_cat[col], "real")
            synth_table = ustats.frequency_table(df_synth_cat[col], "synth")

            # Computes the frequency coverage
            # 1 - average of the frequency differences between real and synthetic
            # TODO: new formula for frequency coverage to penalize it more?
            df_counts = (  # account only for the real dataset categories
                pd.merge(
                    real_table,
                    synth_table,
                    how="left",
                    left_index=True,
                    right_index=True,
                )
                .fillna(0)
                .assign(freq_diff=lambda df: (df["synth_freq"] - df["real_freq"]).abs())
            )

            support_coverage[col] = np.count_nonzero(df_counts["synth_freq"]) / len(
                df_counts
            )

            frequency_coverage[col] = 1 - df_counts["freq_diff"].mean()

            # for detailed results
            real_counts[col] = real_table["real_count"]
            synth_counts[col] = synth_table["synth_count"]

        res = {
            "average": {
                "support_coverage": np.mean(list(support_coverage.values())),
                "frequency_coverage": np.mean(list(frequency_coverage.values())),
            },
            "detailed": {"real_counts": real_counts, "synthetic_counts": synth_counts},
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot for each categorical variables of both real and synthetic datasets.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(key in report for key in ["real_counts", "synthetic_counts"])

        # One plot per variable
        num_rows = len(report["real_counts"])
        max_plot_per_window = 5
        num_win = num_rows // max_plot_per_window
        if num_rows % max_plot_per_window != 0:
            num_win += 1

        axes = []
        for f in range(num_win):
            fig, axes_f = plt.subplots(
                nrows=max_plot_per_window, figsize=figsize, layout="constrained"
            )
            fig.suptitle(f"Metric: {cls.name} ({f + 1}/{num_win})")
            axes.extend(axes_f)

        for i, col in enumerate(report["real_counts"]):
            udraw.bar_plot_hue(
                s=report["real_counts"][col],
                s_nested=report["synthetic_counts"][col],
                original_name="Real",
                nested_name="Synthetic",
                hue_name="Data",
                title=col,
                value_name="Count",
                orient="v",
                xrotation=True,
                ax=axes[i],
            )
            if i % max_plot_per_window != 0:
                axes[i].get_legend().remove()


class UnivariateDiscreteDistance(Metric, metaclass=ABCMeta):
    """
    Check the similarities between real and synthetic data with discrete distance.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "Univariate Distance"
    alias = "univ_dist"
    distance_name: str

    @classmethod
    @property
    @abstractmethod
    def distance_name(cls) -> str:
        """
        :return: the name of the distance computed by the metric
        """

    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the *Hellinger* distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the *Hellinger* distance
        """

        assert len(p) == len(q), "p and q must have the same length"
        assert abs(np.sum(p) - 1) < 0.01, "p must be discrete probability distribution"
        assert abs(np.sum(q) - 1) < 0.01, "q must be discrete probability distribution"

        hell = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

        return hell

    @staticmethod
    def kullback_leibler_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the *Kullback-Leibler* divergence of p from q where p and q are two probability distributions.

        :param p: a probability distribution to compare
        :param q: the probability distribution to be compared to p
        :return: the *Kullback-Leibler* divergence
        """

        assert len(p) == len(q), "p and q must have the same length"
        assert abs(np.sum(p) - 1) < 0.01, "p must be discrete probability distribution"
        assert abs(np.sum(q) - 1) < 0.01, "q must be discrete probability distribution"

        kl_div = scipy.special.rel_entr(p, q)

        return np.ndarray.sum(kl_div)

    @staticmethod
    @abstractmethod
    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the distance
        """
        pass

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        train_test_ref: bool = False,
    ) -> dict:
        """
        Measure the discrete distance between the real and the synthetic data for each variable.
        All variables must be categorical.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :param train_test_ref: a boolean parameter indicating whether the metric is calculated for synthetic data
          or for the test set as a reference. It triggers or not the consistency check on the length of the sets.
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average distance
              between the real and synthetic data across all variables
            * **detailed** -- a dictionary with the variables names as keys
              and the distance between the real and synthetic data
        """

        super().check_consistency_compute_parameters(
            df_real, df_synthetic, metadata, train_test_ref=train_test_ref
        )

        distance = {}
        for col in df_real["train"].columns:
            real_table = ustats.frequency_table(df_real["train"][col], "real")
            synth_table = ustats.frequency_table(df_synthetic["train"][col], "synth")
            # Account for all categories since the distance is defined for
            # probability distributions summing up to 1
            df_counts = pd.merge(
                real_table, synth_table, how="outer", left_index=True, right_index=True
            ).fillna(0)
            dist = self.compute_distance(
                df_counts["synth_freq"].to_numpy(), df_counts["real_freq"].to_numpy()
            )
            distance[col] = dist

        res = {
            "average": {
                f"{self.__class__.distance_name}": np.mean(list(distance.values())),
            },
            "detailed": {f"{self.__class__.distance_name}": distance},
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot of the distance between real and synthetic variables.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert f"{cls.distance_name}" in report

        info = cls.get_average_submetrics()[0]
        assert info["submetric"] == cls.distance_name

        plt.figure(figsize=figsize, layout="constrained")

        udraw.bar_plot(
            data=pd.DataFrame([report[f"{cls.distance_name}"]]),
            title=f"Metric: {cls.name}",
            value_name=f"{cls.distance_name}",
            orient="v",
            lim=(info["min"], info["max"]),
        )


class ContinuousUnivariateDistance(UnivariateDiscreteDistance, metaclass=ABCMeta):
    """
    Check the similarities between real and synthetic data with discrete distance
    for binned continuous variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "Continuous " + UnivariateDiscreteDistance.name
    alias = "cont_" + UnivariateDiscreteDistance.alias

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Estimate the distance between the real and the synthetic data
        for each continuous variable by binning them.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average distance between the real and synthetic data
              across all continuous binned variables
            * **detailed** -- a dictionary with the variables names as keys
              and the distance between the real and synthetic data
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for continuous variables only
        df_real_cont = df_real["train"].drop(columns=metadata["categorical"])
        df_real_cont_test = df_real["test"].drop(columns=metadata["categorical"])
        df_synth_cont = df_synthetic["train"].drop(columns=metadata["categorical"])
        metadata_cont = {
            "continuous": metadata["continuous"],
            "categorical": [],
            "variable_to_predict": None,
        }

        if df_real_cont.shape[1] == 0:
            return {}

        df_real_bin, df_synthetic_bin = upreprocessing.bin_per_column(
            df_ref=df_real_cont, df_tobin=df_synth_cont, bin_size=5
        )

        df_real_bin_ref, df_real_test_bin_ref = upreprocessing.bin_per_column(
            df_ref=df_real_cont, df_tobin=df_real_cont_test, bin_size=5
        )

        df_real_bin = {"train": df_real_bin, "test": None}
        df_synthetic_bin = {"train": df_synthetic_bin, "test": None}
        df_ref_test_bin = {"train": df_real_test_bin_ref, "test": None}

        res = super().compute(df_real_bin, df_synthetic_bin, metadata_cont)
        res["average"][
            f"{self.__class__.distance_name}_train_test_ref"
        ] = super().compute(
            df_real_bin, df_ref_test_bin, metadata_cont, train_test_ref=True
        )[
            "average"
        ][
            f"{self.__class__.distance_name}"
        ]
        res["detailed"][
            f"{self.__class__.distance_name}_train_test_ref"
        ] = super().compute(
            df_real_bin, df_ref_test_bin, metadata_cont, train_test_ref=True
        )[
            "detailed"
        ][
            f"{self.__class__.distance_name}"
        ]

        return res


class CategoricalUnivariateDistance(UnivariateDiscreteDistance, metaclass=ABCMeta):
    """
    Check the similarities between real and synthetic data with discrete distance for categorical variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "Categorical " + UnivariateDiscreteDistance.name
    alias = "cat_" + UnivariateDiscreteDistance.alias

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Measure the discrete distance between the real and the synthetic data for each variable.
        All variables must be categorical.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average distance between the real and synthetic data across all categorical variables
            * **detailed** -- a dictionary with the variables names as keys
              and the distance between the real and synthetic data
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Defined for categorical variables only
        df_real_cat = df_real["train"].drop(columns=metadata["continuous"])
        df_real_test_cat = df_real["test"].drop(columns=metadata["continuous"])
        df_synth_cat = df_synthetic["train"].drop(columns=metadata["continuous"])
        metadata_cat = {
            "continuous": [],
            "categorical": metadata["categorical"],
            "variable_to_predict": None,
        }

        if df_real_cat.shape[1] == 0:
            return {}

        df_real_cat = {"train": df_real_cat, "test": None}
        df_real_test_cat = {"train": df_real_test_cat, "test": None}
        df_synth_cat = {"train": df_synth_cat, "test": None}

        res = super().compute(df_real_cat, df_synth_cat, metadata_cat)
        res["average"][
            f"{self.__class__.distance_name}_train_test_ref"
        ] = super().compute(
            df_real_cat, df_real_test_cat, metadata_cat, train_test_ref=True
        )[
            "average"
        ][
            f"{self.__class__.distance_name}"
        ]
        res["detailed"][
            f"{self.__class__.distance_name}_train_test_ref"
        ] = super().compute(
            df_real_cat, df_real_test_cat, metadata_cat, train_test_ref=True
        )[
            "detailed"
        ][
            f"{self.__class__.distance_name}"
        ]

        return res


class ContinuousUnivariateHellingerDistance(ContinuousUnivariateDistance):
    """
    Check the similarities between real and synthetic data with discrete *Hellinger* distance
    for binned continuous variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "Hellinger " + ContinuousUnivariateDistance.name
    alias = "hell_" + ContinuousUnivariateDistance.alias
    distance_name = "hellinger_distance"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": cls.distance_name,
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "The mean Hellinger distance assesses the difference in distribution between the real and synthetic datasets for each continuous variable.",
                "interpretation": "The smaller the H value, the closer the synthetic dataset is to the real dataset. It can be compared to the distance between the train and test sets - generally, below 0.1 indicates a similar distribution.",
            },
            {
                "submetric": cls.distance_name + "_train_test_ref",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "Reference value for the mean continuous Hellinger distance.",
                "interpretation": "",
            },
        ]

    @staticmethod
    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the distance
        """
        return UnivariateDiscreteDistance.hellinger_distance(p, q)


class CategoricalUnivariateHellingerDistance(CategoricalUnivariateDistance):
    """
    Check the similarities between real and synthetic data with discrete *Hellinger* distance
    for categorical variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "Hellinger " + CategoricalUnivariateDistance.name
    alias = "hell_" + CategoricalUnivariateDistance.alias
    distance_name = "hellinger_distance"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": cls.distance_name,
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "The mean Hellinger distance assesses the difference in distribution between the real and synthetic datasets for each categorical variable.",
                "interpretation": "The smaller the H value, the closer the synthetic dataset is to the real dataset. It can be compared to the distance between the train and test sets - generally, below 0.1 indicates a similar distribution.",
            },
            {
                "submetric": cls.distance_name + "_train_test_ref",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "Reference value for the mean categorical Hellinger distance.",
                "interpretation": "",
            },
        ]

    @staticmethod
    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the distance
        """
        return UnivariateDiscreteDistance.hellinger_distance(p, q)


class ContinuousUnivariateKLDivergence(ContinuousUnivariateDistance):
    """
    Check the similarities between real and synthetic data with discrete *Kullback-Leibler* divergence
    for binned continuous variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "KL Divergence " + ContinuousUnivariateDistance.name
    alias = "kl_div_" + ContinuousUnivariateDistance.alias
    distance_name = "kl_divergence"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": cls.distance_name,
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "The KL divergence is a measure of relative entropy between the synthetic vs. real probability distributions for continuous variables.",
                "interpretation": "It can be compared to the distance between the train and test sets.",
            },
            {
                "submetric": cls.distance_name + "_train_test_ref",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "Reference value for the continuous KL divergence.",
                "interpretation": "",
            },
        ]

    @staticmethod
    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the distance
        """
        return UnivariateDiscreteDistance.kullback_leibler_divergence(p, q)


class CategoricalUnivariateKLDivergence(CategoricalUnivariateDistance):
    """
    Check the similarities between real and synthetic data with discrete *Kullback-Leibler* divergence
    for categorical variables.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar distance_name: the name of the distance to use
    :vartype distance_name: str

    :param random_state: for reproducibility purposes
    """

    name = "KL Divergence " + CategoricalUnivariateDistance.name
    alias = "kl_div_" + CategoricalUnivariateDistance.alias
    distance_name = "kl_divergence"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": cls.distance_name,
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "This KL divergence is a measure of relative entropy between the synthetic vs. real probability distributions for categorical variables.",
                "interpretation": "It can be compared to the distance between the train and test sets.",
            },
            {
                "submetric": cls.distance_name + "_train_test_ref",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "Reference value for the categorical KL divergence.",
                "interpretation": "",
            },
        ]

    @staticmethod
    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the distance between p and q where p and q are two probability distributions.

        :param p: a probability distribution
        :param q: another probability distribution
        :return: the distance
        """
        return UnivariateDiscreteDistance.kullback_leibler_divergence(p, q)
