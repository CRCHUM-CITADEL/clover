# Standard library
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union
from pathlib import Path
import random
from inspect import getfullargspec

# 3rd party packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Local packages
import utils.draw as udraw


class Report(metaclass=ABCMeta):
    """
    Create a report of the metrics.

    :cvar metrics_mapping: the dictionary associating the name of the metric to its class
    :vartype metrics_mapping: dict

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    """

    metrics_mapping: dict

    @classmethod
    @property
    @abstractmethod
    def metrics_mapping(cls) -> dict:
        """
        :return: the dictionary associating the name of the metric to its class
        """

    def __init__(
        self,
        dataset_name: str,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
    ):
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

        # Metrics
        self._metrics = []

        # Metrics results
        self._metrics_info = []
        self._metrics_results = {"average": {}, "detailed": {}}

        # Size of the figures
        self._figsize = {metric_name: figsize for metric_name in self.metrics_mapping}

    def _init_metrics(self, metrics: List[str] = None, params: dict = None) -> None:
        """
        Populate the list of metrics.

        :param metrics: list of the metrics to compute. If not specified, all the metrics are computed.
        :param params: the dictionary of parameters to instantiate the metrics
        :return: *None*
        """
        if metrics is not None:
            assert set(metrics) <= set(self.metrics_mapping.keys()), (
                "Wrong metrics name. Must be among the following list: '"
                + "', '".join(list(self.metrics_mapping.keys()))
                + "'"
            )

        parameters = params if params is not None else {}

        for metric_name in metrics if metrics is not None else self.metrics_mapping:
            args = getfullargspec(self.metrics_mapping[metric_name]).args[
                1:
            ]  # remove self
            metric = self.metrics_mapping[metric_name](
                *[parameters[arg] for arg in args]
            )
            self._metrics.append(metric)

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
            self._metrics_info.append(metric.get_submetrics_info())

            res = metric.compute(self._df_real, self._df_synthetic, self._metadata)

            # Append the submetrics name and value as a dict
            if len(res) != 0:
                for report_type in ["average", "detailed"]:
                    if res[report_type] is not None:
                        self._metrics_results[report_type][class_vars["name"]] = res[
                            report_type
                        ]

        self._metrics_info = pd.concat(self._metrics_info)

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

        df = pd.merge(
            self._metrics_info, df_res, on=["name", "submetric"], how="inner"
        )[["name", "submetric", "value", "objective", "min", "max"]]
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

        assert metric_name in self.metrics_mapping, (
            "Wrong metric name. Must be among the following list: '"
            + "', '".join(list(self.metrics_mapping.keys()))
            + "'"
        )
        assert (
            metric_name in self._metrics_results["detailed"]
        ), "The report does not contain any value for the specified metric"

        fig_size = figsize if figsize is not None else self._figsize[metric_name]

        self.metrics_mapping[metric_name].draw(
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
