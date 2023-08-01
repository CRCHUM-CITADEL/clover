from abc import ABC
from typing import Union  # standard library

import pandas as pd  # 3rd party packages
from pathlib import Path

from .base import Generator  # local
import utils.standard as ustandard
from generators.external.ctabgan.ctabgan_synthesizer import CTABGANSynthesizer
from .external.ctabgan.data_preparation import DataPrep


class CTABGANGenerator(Generator):
    """
    Wrapper for the GAN-based synthesizer presented in the paper CTAB-GAN+: Enhancing Tabular Data Synthesis, by
    Zhao & al.
    https://github.com/Team-TUD/CTAB-GAN-Plus

    See article for more information:
    https://arxiv.org/abs/2204.00401

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    These should comprehend all the columns to synthesize, including the columns in "mixed", "log", and
    "integer".
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param mixed_columns: dictionary of "mixed" column names with corresponding categorical modes
    :param log_columns: list of skewed exponential numerical columns
    :param integer_columns: list of numeric columns without floating numbers
    :param problem_type: dictionary where the key is "Regression", "Classification" or None, and the value is
    the target column. The dimensions of the target column actually determines the loss function (CrossEntropy,
    BinaryCrossEntropy or SmoothL1).
    :param  class_dim: size of each desired linear layer for the auxiliary classifier
    :param random_dim: dimension of the noise vector fed to the generator
    :param num_channels: number of channels in the convolutional layers of both the generator and the discriminator
    :param l2scale: rate of weight decay used in the optimizer of the generator, discriminator and auxiliary classifier
    :param batch_size: batch size for training
    :param epochs: number of traianing epochs
    """

    name = "CTABGAN"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        mixed_columns: dict = {},
        log_columns: list = [],
        integer_columns: list = [],
        problem_type: dict = {},
        class_dim: tuple[int, ...] = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 150,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)
        self._extra_metadata = {
            "mixed_columns": mixed_columns,
            "log_columns": log_columns,
            "integer_columns": integer_columns,
        }
        self._problem_type = problem_type
        self._data_prep = None

        self._params = {
            "class_dim": class_dim,
            "random_dim": random_dim,
            "num_channels": num_channels,
            "l2scale": l2scale,
            "batch_size": batch_size,
            "epochs": epochs,
        }

        self._gen = (
            None
            if generator_filepath is None
            else ustandard.load_pickle(filepath=generator_filepath)
        )

    def preprocess(self) -> None:
        """
        Creation of the DataPrep object from the CTAB-GAN plus code. This is used for both pre-processing
        of the original data and postprocessing of the generated data.
        """

        self._data_prep = DataPrep(
            raw_df=self._df,
            categorical=self._metadata["categorical"],
            log=self._extra_metadata["log_columns"],
            mixed=self._extra_metadata["mixed_columns"],
            integer=self._extra_metadata["integer_columns"],
            type=self._problem_type,
        )

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        The following arguments could be added to the function (or to the initialization):
        general: a list including columns that should receive the "General Transform" treatment,
        that is single-mode Gaussian variables or categorical variables that contain so many
        categories that the available machines can not train with the encoded data.
        non_categorical: a list including columns that are also in "categorical" but are very high
        dimensional. Columns in "non_categorical_columns" are first encoded to numerical numbers,
        and then treated as continuous columns, using variational gaussian mixture.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._gen = CTABGANSynthesizer(**self._params)
        self._gen.fit(
            train_data=self._data_prep.df,
            categorical=self._data_prep.column_types["categorical"],
            mixed=self._data_prep.column_types["mixed"],
            # general=self._data_prep.column_types["general"],
            # non_categorical=self._data_prep.column_types["non_categorical"],
            type=self._problem_type,
        )

        ustandard.save_pickle(
            obj=self._gen, path=save_path, filename=CTABGANGenerator.name
        )

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the synthesizer trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        sample = self._gen.sample(num_samples)
        sample_df = self._data_prep.inverse_prep(sample)

        return sample_df

    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        print("CTAB-GAN+ Synthesizer parameters: \n")
        for key, value in self._gen.__dict__.items():
            print(str(key) + ": " + str(value))
