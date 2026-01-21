from abc import ABCMeta, abstractmethod  # Standard library
from typing import Union
from pathlib import Path

import pandas as pd  # 3rd party packages

import clover.utils.standard as ustandard  # local


class Generator(metaclass=ABCMeta):
    """
    Abstract class providing the template to follow for each generator.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    """

    name: str

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        """
        :return: the name of the generator
        """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
    ):
        self._check_consistency_init_parameters(df, metadata)
        assert generator_filepath is None or Path(generator_filepath).exists()

        self._df = df
        self._metadata = metadata
        self._random_state = random_state
        self._gen = (
            None
            if generator_filepath is None
            else ustandard.load_pickle(filepath=generator_filepath)
        )

    @abstractmethod
    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """
        pass

    @staticmethod
    def _check_consistency_init_parameters(df: pd.DataFrame, metadata: dict) -> None:
        """
        Assert that the init parameters are consistent.

        :param df: the dataset to synthesize
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: *None*
        """

        assert {"continuous", "categorical", "variable_to_predict"} == set(
            metadata.keys()
        ), "Missing keys in the metadata dictionary"

        assert set(metadata["continuous"] + metadata["categorical"]) == set(
            df.columns
        ), "All columns should be specified in the metadata"

        assert (
            len(metadata["continuous"] + metadata["categorical"]) == df.shape[1]
        ), "All columns should be specified once in the metadata"

        assert (
            metadata["variable_to_predict"] is None
            or metadata["variable_to_predict"] in df.columns
        ), "The variable to predict should be in the dataset"

    @abstractmethod
    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        :param save_path: the path to save the generator
        :return: *None*
        """
        pass

    @abstractmethod
    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        pass

    @abstractmethod
    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the synthesizer trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """
        pass
