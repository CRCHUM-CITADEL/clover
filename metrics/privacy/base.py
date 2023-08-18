from abc import ABCMeta, abstractmethod  # Standard library
from typing import Tuple, List
import random

import pandas as pd  # 3rd party packages
import numpy as np


class PrivacyMetric(metaclass=ABCMeta):
    """
    Abstract privacy metric class providing the template to follow for each metric.

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
    @abstractmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

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
    ) -> None:
        """
        Assert that the compute method parameters are consistent.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: *None*
        """

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
