from typing import Union, List  # standard library
from pathlib import Path
import warnings

from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN  # 3rd party packages
import pandas as pd
import numpy as np

from generators.base import Generator  # local
import utils.standard as ustandard


class SmoteGenerator(Generator):
    """
    Wrapper of the SMOTE (Synthetic Minority Oversampling TEchnique) Python implementation from imbalanced-learn.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param k_neighbors: the number of neighbors used to find the avatar
    """

    name = "SMOTE"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        k_neighbors: int = 5,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._params = self._gen.get_params() if self._gen is not None else None

        self._prediction_type = (
            "Classification"
            if self._metadata["variable_to_predict"] in self._metadata["categorical"]
            else "Regression"
        )

        self._k_neighbors = k_neighbors
        self._contains_cont_indep_vars = None
        self._contains_cat_indep_vars = None

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # Instantiate the SMOTE method according to the variable types
        self._contains_cont_indep_vars = len(self._metadata["continuous"]) > 0
        cat_indep_vars = [
            i
            for i, col in enumerate(self._df.columns)
            if col in self._metadata["categorical"]
            and col != self._metadata["variable_to_predict"]  # dependent variable
        ]
        self._contains_cat_indep_vars = len(cat_indep_vars) > 0

        # Parameters for the SMOTE instantiation
        self._params = {"random_state": self._random_state}
        if self._k_neighbors is not None:
            self._params["k_neighbors"] = self._k_neighbors
        if self._contains_cont_indep_vars and self._contains_cat_indep_vars:
            self._params["categorical_features"] = cat_indep_vars

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Instantiate the SMOTE object if not already loaded.

        :param save_path: the path to save the generator
        :return: *None*
        """

        if self._contains_cont_indep_vars and self._contains_cat_indep_vars:
            self._gen = SMOTENC(**self._params)
        elif self._contains_cont_indep_vars:
            self._gen = SMOTE(**self._params)
        else:
            self._gen = SMOTEN(**self._params)

        ustandard.save_pickle(
            obj=self._gen, folderpath=save_path, filename=SmoteGenerator.name, date=True
        )

    def display(self) -> None:
        """
        Print the SMOTE method and its parameters.

        :return: *None*
        """

        print("Oversampling generator: ", type(self._gen))
        print("Parameters: ", self._params)

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the oversampling SMOTE method.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        # Prepare the X and y depending on the prediction type (classification or regression)
        if self._prediction_type == "Classification":
            X = self._df.drop(columns=self._metadata["variable_to_predict"])
            y = self._df[self._metadata["variable_to_predict"]]
        else:
            # SMOTE is not defined for regression: we add a fake minority sample so
            # that the neighbors are searched across all majority samples
            X = self._df.copy().reset_index(drop=True)
            X.loc[len(X)] = X.iloc[-1]
            y = np.array([0] * len(self._df) + [1])

        # Number of samples in each class
        # (SMOTE is an oversampling method so the number needs to be superior to the original one)
        num_real_samples = len(self._df)
        sampling_strategy = {}
        if self._prediction_type == "Regression":
            sampling_strategy[1] = 1
            sampling_strategy[0] = len(self._df) + num_samples
        else:  # Classification
            # The ratio of each class must be preserved
            counts = y.value_counts() / num_real_samples
            counts = (counts * num_samples).round().astype(int)
            counts += (
                y.value_counts()
            )  # oversampling, we add new samples to the original ones
            if counts.sum() != num_real_samples + num_samples:
                # we add or remove the extra samples due to the rounding to the most frequent class
                counts.iloc[0] = counts.iloc[0] + (
                    (num_real_samples + num_samples) - counts.sum()
                )
            sampling_strategy = counts.to_dict()

        # Update SMOTE parameters
        self._params["sampling_strategy"] = sampling_strategy
        self._gen.set_params(**self._params)

        # Fit and resample (cannot be separated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_synth, y_synth = self._gen.fit_resample(X, y)

        # Separate the real samples from the synthetic ones
        if self._prediction_type == "Regression":
            samples = X_synth.loc[
                num_real_samples + 1 :, :
            ]  # Real samples plus the fake one - the labels were fake
        else:  # Classification
            samples = pd.concat(  # Merge back the independent variables and the dependent one and remove real samples
                [X_synth.loc[num_real_samples:, :], y_synth.loc[num_real_samples:]],
                axis=1,
            )
        samples = samples.reset_index(drop=True)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{SmoteGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
