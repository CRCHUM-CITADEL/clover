# Standard library
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union
import random
from pathlib import Path
from inspect import getfullargspec

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local packages
import clover.utils.standard as ustandard
import clover.utils.draw as udraw


class Metric(metaclass=ABCMeta):
    """
    Abstract metric class providing the template to follow for each metric.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name: str
    alias: str

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        """
        :return: the name of the metric
        """

    @classmethod
    @property
    @abstractmethod
    def alias(cls) -> str:
        """
        :return: the alias of the metric
        """

    def __init__(self, random_state: int = None):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    @classmethod
    def get_class_variables(cls) -> dict:
        """
        Getter for the class variables.

        :return: a dict containing the name of the class variables as key and their value
        """

        class_variables = {
            "name": cls.name,
            "alias": cls.alias,
        }
        return class_variables

    @classmethod
    def get_submetrics_info(cls) -> pd.DataFrame:
        """
        Get the average submetrics information.

        :return: the submetrics information as a dataframe
        """

        df = pd.DataFrame(cls.get_average_submetrics())
        df["name"] = cls.name
        df["alias"] = cls.alias

        return df

    @classmethod
    @abstractmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max**, **objective**, **description** and **interpretation**)
        """

    @abstractmethod
    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compute the metric. To be reimplemented for each metric.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary containing two keys: the **average** metric values and the **detailed** ones
        """
        pass

    @staticmethod
    def check_consistency_compute_parameters(
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        train_test_ref: bool = False,
    ) -> None:
        """
        Assert that the compute method parameters are consistent.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :param train_test_ref: a boolean parameter indicating whether the metric is calculated for synthetic data
          or for the test set as a reference. It triggers or not the consistency check on the length of the sets.
        :return: *None*
        """

        if train_test_ref is False:
            assert (
                df_real["train"].shape == df_synthetic["train"].shape
            ), "Train sets must have the same shape"
            assert (df_real["test"] is None and df_synthetic["test"] is None) or (
                df_real["test"].shape == df_synthetic["test"].shape
            ), "Test sets must have the same shape"

        assert set(df_real["train"].columns) == set(
            df_synthetic["train"].columns
        ), "Train sets must have the same columns"

        assert df_real["test"] is None or set(df_real["test"].columns) == set(
            df_synthetic["test"].columns
        ), "Test sets must have the same columns"

        assert {"continuous", "categorical", "variable_to_predict"} == set(
            metadata.keys()
        ), "Missing keys in the metadata dictionary"

        assert set(metadata["continuous"] + metadata["categorical"]) == set(
            df_real["train"].columns
        ), "All columns should be specified in the metadata"

        assert (
            len(metadata["continuous"] + metadata["categorical"])
            == df_real["train"].shape[1]
        ), "All columns should be specified once in the metadata"

        assert (
            metadata["variable_to_predict"] is None
            or metadata["variable_to_predict"] in df_real["train"].columns
        ), "The variable to predict should be in the dataset"

    @classmethod
    @abstractmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Create a graphical visualization of the metric based on the detailed report.
        To be reimplemented for each metric.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        pass


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
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param report_filepath: the path of a computed report if available
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
        dataset_name: str = None,
        df_real: dict[str, pd.DataFrame] = None,
        df_synthetic: dict[str, pd.DataFrame] = None,
        metadata: dict = None,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
        report_filepath: Union[Path, str] = None,
    ):
        assert report_filepath is None or Path(report_filepath).exists()
        assert report_filepath is not None or all(
            v is not None for v in [dataset_name, df_real, df_synthetic, metadata]
        )

        # Seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Datasets
        self._dataset_name = dataset_name
        self._metadata = metadata
        self._df_real = df_real
        self._df_synthetic = df_synthetic
        self._num_instances = (
            [len(df_real["train"]), len(df_real["test"])]
            if df_real is not None
            else [0, 0]
        )
        self._num_variables = df_real["train"].shape[1] if df_real is not None else 0
        self._num_continuous_variables = (
            len(metadata["continuous"]) if metadata is not None else 0
        )
        self._num_categorical_variables = (
            len(metadata["categorical"]) if metadata is not None else 0
        )

        # Metrics
        self._metrics = []

        # Metrics results
        if report_filepath is None:
            self._metrics_results = {"average": {}, "detailed": {}}
        else:
            self._metrics_results = ustandard.load_pickle(report_filepath)

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

    @classmethod
    def get_metrics_info(cls) -> pd.DataFrame:
        """
        Get the average submetrics information for all the metrics.

        :return: the submetrics information as a dataframe
        """

        info = []

        for metric in cls.metrics_mapping:
            info.append(cls.metrics_mapping[metric].get_submetrics_info())

        info = pd.concat(info, axis=0)

        return info

    def compute(self) -> None:
        """
        Compute all metrics one by one and store the resulting dictionaries.

        :return: *None*
        """

        assert len(self._metrics) > 0, "No metric to compute"

        # Compute the results for each metric and store them
        for metric in self._metrics:
            class_vars = metric.get_class_variables()

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

    def save(self, savepath: Union[Path, str], filename: str) -> None:
        """
        Pickle the report.

        :param savepath: the save folder
        :param filename: the filename without the extension
        :return: *None*
        """
        ustandard.save_pickle(self._metrics_results, savepath, filename)

    def summary(self) -> pd.DataFrame:
        """
        Report the average utility metrics values across all variables,
        distinguishing continuous variables from discrete ones.

        :return: a pandas dataframe
        """

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

        info = self.get_metrics_info()
        df = pd.merge(info, df_res, on=["name", "submetric"], how="inner")[
            ["name", "alias", "submetric", "value", "objective", "min", "max"]
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
