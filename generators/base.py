from abc import ABCMeta, abstractmethod  # Standard library
from typing import Union
from pathlib import Path

import pandas as pd  # 3rd party packages


class Generator(metaclass=ABCMeta):
    """
    Abstract class providing the template to follow for each generator.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
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
        generator_filepath: Union[Path, str] = None,
    ):
        self._df = df
        self._metadata = metadata
        self._generator_filepath = generator_filepath

    @abstractmethod
    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """
        pass

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

    @abstractmethod
    def search_hyperparameters(self, **kwargs) -> dict:
        """
        Find the best hyperparameters for the generator.

        :param kwargs: a dict containing the parameters of the search algorithm
        :return: a dict with the best hyperparameters
        """
        pass
