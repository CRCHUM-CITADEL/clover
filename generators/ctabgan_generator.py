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
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables.
        These should comprehend all the columns to synthesize, including the columns in "mixed", "log", and
        "integer".
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param mixed_columns: dictionary of "mixed" column names with corresponding categorical modes. Mixed columns are
        mostly continuous columns while one value or more - modes - hold another meaning (ex: 0).
    :param log_columns: list of skewed exponential numerical columns. These columns will go through a log transform.
    :param integer_columns: list of numeric columns without floating numbers. These columns will be rounded in the
        sampling step.
    :param class_dim: size of each desired linear layer for the auxiliary classifier
    :param random_dim: dimension of the noise vector fed to the generator
    :param num_channels: number of channels in the convolutional layers of both the generator and the discriminator
    :param l2scale: rate of weight decay used in the optimizer of the generator, discriminator and auxiliary classifier
    :param batch_size: batch size for training
    :param epochs: number of training epochs
    """

    name = "CTABGAN"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        preprocess_metadata: dict = None,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        mixed_columns: dict = None,
        log_columns: list = None,
        integer_columns: list = None,
        class_dim: tuple[int, ...] = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 150,
        epsilon=None,
        preprocess_epsilon_pp: float = None,
        delta=None,
        max_grad_norm=1,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)
        self._extra_metadata = {
            "mixed_columns": mixed_columns if mixed_columns is not None else {},
            "log_columns": log_columns if log_columns is not None else [],
            "integer_columns": integer_columns if integer_columns is not None else [],
        }

        prediction = (
            "Classification"
            if metadata["variable_to_predict"] in metadata["categorical"]
            else "Regression"
        )
        # The dimensions of the target column actually determines the loss function
        # (CrossEntropy, BinaryCrossEntropy or SmoothL1).
        self._problem_type = {prediction: metadata["variable_to_predict"]}

        self._data_prep = None
        self._preprocess_metadata = preprocess_metadata

        self._params = {
            "class_dim": class_dim,
            "random_dim": random_dim,
            "num_channels": num_channels,
            "l2scale": l2scale,
            "batch_size": batch_size,
            "epochs": epochs,
            "epsilon": epsilon,
            "preprocess_epsilon_pp": preprocess_epsilon_pp,
            "delta": delta,
            "max_grad_norm": max_grad_norm,
        }

        if not (
            (epsilon is None and delta is None)
            or (epsilon is not None and delta is not None)
        ):
            raise ValueError(
                "epsilon and delta should either both be specified for differentially private training, "
                "or none should be for non-DP training"
            )

        assert (
            0 <= preprocess_epsilon_pp <= 1
        ), "preprocess_epsilon must be in the interval [0, 1]"

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

        if self._params["epsilon"] is not None:
            self._gen.fit_dp(
                train_data=self._data_prep.df,
                categorical=self._data_prep.column_types["categorical"],
                mixed=self._data_prep.column_types["mixed"],
                # general=self._data_prep.column_types["general"],
                # non_categorical=self._data_prep.column_types["non_categorical"],
                type=self._problem_type,
                preprocess_metadata=self._preprocess_metadata,
            )

        else:
            self._gen.fit(
                train_data=self._data_prep.df,
                categorical=self._data_prep.column_types["categorical"],
                mixed=self._data_prep.column_types["mixed"],
                # general=self._data_prep.column_types["general"],
                # non_categorical=self._data_prep.column_types["non_categorical"],
                type=self._problem_type,
            )

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=CTABGANGenerator.name,
            date=True,
        )

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the synthesizer trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        samples = self._gen.sample(num_samples)
        samples = self._data_prep.inverse_prep(samples)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{CTABGANGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples

    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        print("CTAB-GAN+ Synthesizer parameters: \n")
        for key, value in self._gen.__dict__.items():
            print(str(key) + ": " + str(value))
