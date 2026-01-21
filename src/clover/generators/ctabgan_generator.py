from typing import Dict, Union  # standard library
import copy

import pandas as pd  # 3rd party packages
import numpy as np
from pathlib import Path
import warnings

from .base import Generator  # local
import clover.utils.standard as ustandard
from clover.utils.postprocessing import transform_data
from clover.generators.external.ctabgan.ctabgan_synthesizer import (
    CTABGANSynthesizer,
)

from .external.ctabgan.data_preparation import DataPrep
from .external.ctabgan.model.synthesizer.transformer import DataTransformer

from clover.utils.preprocessing import generate_continuous_dp


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
    :param epsilon: the privacy budget of the differential privacy.
        One can specify how the budget is split between pre-processing and fitting the model.
        For example, epsilon = {"preprocessing": 0.1, "fitting": 0.9}.
        If a single float number is provide, half the budget is allocated for pre-processing
        and half for fitting.
    :param delta: target delta to be achieved for fitting (for differentially private model)
    :param max_grad_norm: the maximum norm of the per-sample gradients.
        Any gradient with norm higher than this will be clipped to this value. (for differentially private model)
    :param preprocess_metadata: specify the range (minimum and maximum) and optionally num_bins and decimals
        (to generate differentially private continuous samples) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening (for differentially private model).
        For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
    """

    name = "CTABGAN"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
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
        epsilon: Union[float, Dict[str, float]] = None,
        delta: float = None,
        max_grad_norm: float = 1.0,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        # Initiate privacy budget
        if (epsilon is None) or isinstance(epsilon, float) or isinstance(epsilon, int):
            preprocess_epsilon_pp = None
        else:
            if isinstance(epsilon, dict):
                epsilon_total = epsilon["preprocessing"] + epsilon["fitting"]
                preprocess_epsilon_pp = epsilon["preprocessing"] / epsilon_total
                epsilon = epsilon_total

        # Convert the preprocess_metadata to the required format
        preprocess_metadata = copy.deepcopy(preprocess_metadata)
        if preprocess_metadata is not None:
            for col in preprocess_metadata:
                if col in metadata["categorical"]:
                    preprocess_metadata[col] = preprocess_metadata[col]["categories"]
                elif col in metadata["continuous"]:
                    preprocess_metadata[col]["min_val"] = preprocess_metadata[col].pop(
                        "min"
                    )
                    preprocess_metadata[col]["max_val"] = preprocess_metadata[col].pop(
                        "max"
                    )

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
            preprocess_epsilon_pp is None or 0 <= preprocess_epsilon_pp <= 1
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

        if self._params["epsilon"] is not None:
            train_data = self._data_prep.df
            categorical = self._data_prep.column_types["categorical"]
            mixed = self._data_prep.column_types["mixed"]
            self._generated_metadata = self._preprocess_metadata

            # We need to add DP to the preprocessing steps
            continuous_columns = [
                column
                for column_index, column in enumerate(train_data.columns)
                if column_index not in categorical and column_index not in mixed.keys()
            ]

            if self._params["preprocess_epsilon_pp"] == 0:
                warnings.warn(
                    "preprocess_epsilon_pp was set to 0. No privacy budget will be dedicated to preprocessing. "
                    "It will be based on the properties of the real data, which may not satisfy differential privacy."
                )

            if self._params["preprocess_epsilon_pp"] is None:
                self._params["preprocess_epsilon_pp"] = 0.5
                warnings.warn(
                    "Half of the privacy budget will be used for preprocessing purposes."
                )

            if self._generated_metadata is None:
                warnings.warn(
                    "Metadata for preprocessing was not provided. It will be generated based on the properties "
                    "of the real data, which may not satisfy differential privacy."
                )
                self._generated_metadata = {}

            for col in train_data.columns:
                col_index = train_data.columns.get_loc(col)
                if (
                    col not in self._generated_metadata.keys()
                    and col in continuous_columns
                ):
                    warnings.warn(
                        f"Metadata for preprocessing {col} was not provided. It will be generated based on the properties "
                        "of the real data, which may not satisfy differential privacy."
                    )
                    self._generated_metadata[col] = {
                        "min_val": np.min(train_data[col]),
                        "max_val": np.max(train_data[col]),
                    }
                elif (
                    col not in self._generated_metadata.keys()
                    and col_index in mixed.keys()
                ):
                    warnings.warn(
                        f"Metadata for preprocessing {col} was not provided. It will be generated based on the properties "
                        "of the real data, which may not satisfy differential privacy."
                    )
                    self._generated_metadata[col] = {
                        "min_val": np.min(
                            train_data[~train_data[col].isin(mixed[col])][col]
                        ),
                        "max_val": np.max(
                            train_data[~train_data[col].isin(mixed[col])][col]
                        ),
                    }
                elif (
                    col in self._generated_metadata.keys() and col in continuous_columns
                ):
                    assert (
                        "min_val" in self._generated_metadata[col].keys()
                        and "max_val" in self._generated_metadata[col].keys()
                    ), (
                        f"Both the minimum and maximum values of {col} need to be specified in the metadata for "
                        f"preprocessing."
                    )
                if (
                    col not in self._generated_metadata.keys()
                    and col_index in categorical
                ):
                    self._generated_metadata[col] = train_data[col].unique()
                elif (
                    col in self._generated_metadata.keys() and col_index in categorical
                ):
                    assert isinstance(
                        self._generated_metadata[col], list
                    ) or isinstance(self._generated_metadata[col], np.ndarray), (
                        f"{col} is a categorical variable. The metadata for preprocessing should be a list of unique "
                        f"categories."
                    )

            preprocess_data = train_data.copy()
            preprocess_data = pd.concat([preprocess_data] * 100, ignore_index=True)
            # Each category of columns needs its own mechanism
            # 1. Columns that are in continuous (metadata) but not in mixed (extra_metadata) should be synthesized like
            # they are in CTGAN
            if self._params["preprocess_epsilon_pp"] > 0:
                for col in continuous_columns:
                    preprocess_data[col] = generate_continuous_dp(
                        df=preprocess_data,
                        col=col,
                        epsilon=self._params["epsilon"]
                        * self._params["preprocess_epsilon_pp"]
                        / len(continuous_columns + list(mixed.keys())),
                        sensitivity=1,
                        **self._generated_metadata[col],
                    )
                # 2. Columns that are in mixed (extra_metadata) should only be synthesized on the values that are not
                # equal to the specified modes
                for col in mixed.keys():
                    preprocess_data[col] = generate_continuous_dp(
                        df=preprocess_data,
                        col=col,
                        epsilon=self._params["epsilon"]
                        * self._params["preprocess_epsilon_pp"]
                        / len(continuous_columns + list(mixed.keys())),
                        sensitivity=1,
                        modes=mixed[col],
                        **self._generated_metadata[col],
                    )
            # 3. Categorical columns should be able to receive a specified list of categories
            # This will be specified directly in DataTransformer through the argument categories.
            if len(categorical) > 0:
                categories_dict = {
                    train_data.columns[column_index]: self._generated_metadata[
                        train_data.columns[column_index]
                    ]
                    for column_index in categorical
                }
            else:
                categories_dict = None

            self.transformer = DataTransformer(
                train_data=preprocess_data,
                categorical_list=categorical,
                mixed_dict=mixed,
                # general_list=self._data_prep.column_types["general"],
                # non_categorical_list=self._data_prep.column_types["non_categorical"],
                categories_dict=categories_dict,
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

        with ustandard.HiddenPrints():  # turn off the prints
            if self._params["epsilon"] is not None:
                self._gen.fit_dp(
                    train_data=self._data_prep.df,
                    type=self._problem_type,
                    transformer=self.transformer,
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

    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        print("CTAB-GAN+ Synthesizer parameters: \n")
        for key, value in self._gen.__dict__.items():
            print(str(key) + ": " + str(value))

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the synthesizer trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        samples = self._gen.sample(num_samples)
        samples = self._data_prep.inverse_prep(samples)

        # Post-processing
        samples = transform_data(
            df_ref=self._df,
            df_to_trans=samples,
            cont_col=self._metadata["continuous"],
        )

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{CTABGANGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
