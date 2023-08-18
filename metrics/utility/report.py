from typing import List, Tuple, Union  # standard library
from pathlib import Path
import random
from inspect import getfullargspec

import pandas as pd  # 3rd party packages
import matplotlib.pyplot as plt
import numpy as np

import metrics.utility.univariate as uni  # local packages
import metrics.utility.bivariate as biv
import metrics.utility.population as pop
import metrics.utility.application as app
import utils.draw as udraw


metrics_mapping = {  # the dictionary associating the name of the metric to its class
    m.name: m
    for m in uni.get_metrics()
    + biv.get_metrics()
    + app.get_metrics()
    + pop.get_metrics()
}


class Report:
    """
    Create a report of the utility metrics.

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
    :param cross_learning: the Cross Learning metrics can slown down the report computation.
        Set to *False* to exclude these metrics. Not taken into account if a list of metrics is provided.
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    :param alpha: the significance level for the chi square test
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    """

    def __init__(
        self,
        dataset_name: str,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        metrics: List[str] = None,
        cross_learning: bool = True,
        num_repeat: int = 20,
        num_folds: int = 10,  # TODO: add a loop
        use_gpu: bool = False,
        alpha: float = 0.05,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
    ):
        if metrics is not None:
            assert set(metrics) <= set(metrics_mapping.keys()), (
                "Wrong metrics name. Must be among the following list: '"
                + "', '".join(list(metrics_mapping.keys()))
                + "'"
            )

        # Seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Datasets
        self._dataset_name = dataset_name
        self._metadata = metadata
        self._df_real = df_real
        self._df_synthetic = df_synthetic
        self._num_instances = [len(df_real["train"]), len(df_real["test"])]
        self._num_variables = df_real["train"].shape[1]
        self._num_continuous_variables = len(metadata["continuous"])
        self._num_categorical_variables = len(metadata["categorical"])

        # Metrics instantiation with their respective parameters
        params = {
            "random_state": None,
            "alpha": alpha,
            "num_repeat": num_repeat,
            "use_gpu": use_gpu,
        }

        self._metrics = []
        for metric_name in metrics if metrics is not None else metrics_mapping:
            args = getfullargspec(metrics_mapping[metric_name]).args[1:]  # remove self
            metric = metrics_mapping[metric_name](*[params[arg] for arg in args])
            self._metrics.append(metric)

        # Remove Cross Learning metrics that can slow down the report computation
        if metrics is None and not cross_learning:
            self._metrics = [
                metric
                for metric in self._metrics
                if not isinstance(metric, pop.CrossLearning)
            ]

        # Metrics results
        self._metrics_info = {}
        self._metrics_results = {"average": {}, "detailed": {}}

        # Size of the figures
        self._figsize = {metric_name: figsize for metric_name in metrics_mapping}
        figsize_longer = (figsize[0], figsize[1] * 1.5)
        figsize_larger = (figsize[0] * 1.5, figsize[1])
        self._figsize[uni.ContinuousStatistics.name] = figsize_larger
        self._figsize[uni.CategoricalStatistics.name] = figsize_longer
        self._figsize[biv.PairwiseCorrelationDifference.name] = figsize_larger
        self._figsize[pop.CrossRegression.name] = figsize_larger
        self._figsize[pop.CrossClassification.name] = figsize_larger

    def get_num_continuous_variables(self) -> int:
        """
        Getter.

        :return: the number of continuous variables in the datasets
        """
        return self._num_continuous_variables

    def get_num_categorical_variables(self) -> int:
        """
        Getter.

        :return: the number of categorical variables in the datasets
        """
        return self._num_categorical_variables

    def compute(self) -> None:
        """
        Compute all metrics one by one and store the resulting dictionaries.

        :return: *None*
        """

        # Compute the results for each metric and store them
        for metric in self._metrics:
            class_vars = metric.get_class_variables()
            self._metrics_info[class_vars["name"]] = class_vars

            res = metric.compute(self._df_real, self._df_synthetic, self._metadata)

            # Append the submetrics name and value as a dict
            if len(res) != 0:
                for report_type in ["average", "detailed"]:
                    if res[report_type] is not None:
                        self._metrics_results[report_type][class_vars["name"]] = res[
                            report_type
                        ]

    def specification(self) -> None:
        """
        Print the dataset specification.

        :return: *None*
        """
        print(f"----- {self._dataset_name} -----")
        print("Contains:")
        print(f"    - {self._num_instances[0]} instances in the train set,")
        print(f"    - {self._num_instances[1]} instances in the test set,")
        print(
            f"    - {self._num_variables} variables, "
            f"{self._num_continuous_variables} continuous and "
            f"{self._num_categorical_variables} categorical."
        )

    def summary(self) -> pd.DataFrame:
        """
        Report the average utility metrics values across all variables,
        distinguishing continuous variables from discrete ones.

        :return: a pandas dataframe
        """

        assert len(self._metrics_info) != 0, (
            "The compute method needs to be called to "
            "obtain the results of the metrics to be summarized"
        )

        # Convert the info dict to a pandas dataframe
        df_info = pd.DataFrame.from_dict(
            self._metrics_info, orient="index"
        ).reset_index(drop=True)

        # Convert the average results from wide to long format
        df_res = (
            pd.DataFrame.from_dict(self._metrics_results["average"], orient="index")
            .rename_axis("name")
            .reset_index()
            .melt(id_vars="name", var_name="submetric", value_name="value")
            .dropna(
                axis="index", how="any"
            )  # since the metrics do not have the same submetrics TODO: find a better way?
        )

        df = pd.merge(df_info, df_res, on="name", how="inner")[
            ["name", "submetric", "value", "objective", "min", "max"]
        ]
        return df

    def detailed(
        self,
        show: bool = True,
        save_folder: Union[str, Path] = None,
        figure_format: str = "pdf",
    ) -> None:
        """
        Detailed graphical visualisation of the utility metrics.

        :param show: display the plots one at a time
        :param save_folder: the path of the folder to save the figure if needed
        :param figure_format: the format of the figure
        :return: *None*
        """

        assert len(self._metrics_info) != 0, (
            "The compute method needs to be called to "
            "obtain the results of the metrics to be summarized"
        )

        for metric_name in self._metrics_results["detailed"]:
            self.draw(
                metric_name=metric_name,
                figsize=self._figsize[metric_name],
                show=show,
                save_folder=save_folder,
                figure_format=figure_format,
            )

    def draw(
        self,
        metric_name,
        figsize: Tuple[float, float] = None,
        show: bool = True,
        save_folder: Union[str, Path] = None,
        figure_format: str = "pdf",
    ) -> None:
        """
        Detailed graphical visualisation of the specified utility metric.

        :param metric_name: the name of the metric to plot
        :param figsize: the size of the figure in inches (width, height)
        :param show: display the plot
        :param save_folder: the path of the folder to save the figure if needed
        :param figure_format: the format of the figure
        :return: *None*
        """

        assert metric_name in metrics_mapping, (
            "Wrong metric name. Must be among the following list: '"
            + "', '".join(list(metrics_mapping.keys()))
            + "'"
        )
        assert (
            metric_name in self._metrics_results["detailed"]
        ), "The report does not contain any value for the specified metric"

        fig_size = figsize if figsize is not None else self._figsize[metric_name]

        metrics_mapping[metric_name].draw(
            report=self._metrics_results["detailed"][metric_name],
            figsize=fig_size,
        )

        if save_folder is not None:
            udraw.save_figure(
                save_folder=save_folder,
                filename=metric_name,
                figure_format=figure_format,
            )

        if show:
            plt.show()
        else:
            plt.close("all")
