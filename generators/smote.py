from typing import Union, List  # standard library
from pathlib import Path
import warnings

from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN  # 3rd party packages
from sklearn.preprocessing import MinMaxScaler
from diffprivlib.utils import PrivacyLeakWarning
import pandas as pd
import numpy as np

from generators.base import Generator  # local
from generators.models.dpsmote import DPSmote
import utils.standard as ustandard


class SmoteGenerator(Generator):
    """
    Wrapper of the SMOTE (Synthetic Minority Oversampling Technique) Python implementation from imbalanced-learn
    and a differentially private version for continuous features.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param epsilon: the privacy budget of the differential privacy.
        If epsilon is set to None, a non-DP model will be trained
    :param k_neighbors: the number of neighbors used to find the avatar (applicable to non-DP generator)
    :param l_connectivity: the distance to decide the neighborhood (applicable to DP generator)
    :param nu: the granularity parameter of the uniform grid that the data will be partitioned into
        (applicable to DP generator)
    :param bounds: specify the range (minimum and maximum) for all numerical columns.
        This ensures that no further privacy leakage is happening. For example,
        bounds = {"col1": {"min": 0, "max": 1}}. If not specified, they will be estimated from the real data
        and a warning will be raised (applicable to DP generator)
    :param r: the range each feature will fall into after preprocessing, i.e., [-r, r] (applicable to DP generator)
    """

    name = "SMOTE"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        epsilon: float = None,
        k_neighbors: int = 5,
        l_connectivity: int = 2,
        nu: float = 0.25,
        bounds: dict = None,
        r: float = 1,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self.epsilon = epsilon
        self._params = self._gen.get_params() if self._gen is not None else None

        self._prediction_type = (
            "Classification"
            if self._metadata["variable_to_predict"] in self._metadata["categorical"]
            else "Regression"
        )

        if self.epsilon is None:  # Initiate non-DP generator
            self._k_neighbors = k_neighbors
            self._contains_cont_indep_vars = None
            self._contains_cat_indep_vars = None
        else:  # Initiate DP generator
            self.l_connectivity = l_connectivity
            self.nu = nu
            self.r = r
            self._original_dtypes = df.dtypes.to_dict()

            # bound dict
            if bounds is None:
                bounds = {}
            self.bounds = bounds

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        if self.epsilon is None:
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
        else:  # DP-SMOTE: Rescale features so that each feature contains entries in the same range, i.e., [-r, r]
            self.encoders_priv = (
                {}
            )  # Encoder to scale the range of each variable to provided bounds and no decoding should be performed
            self.encoders = {}
            self.df_transformed = self._df[
                self._metadata["continuous"]
            ]  # Initiate the transformed data

            # Rescale each variable to the provided bounds to prevent privacy leakage in decoding stage
            for col, series in self.df_transformed.items():
                if col not in self.bounds:
                    warnings.warn(
                        f"upper and lower bounds not specified for column '{col}'",
                        PrivacyLeakWarning,
                    )
                    self.bounds[col] = {"min": series.min(), "max": series.max()}
                self.encoders_priv[col] = MinMaxScaler(
                    feature_range=(self.bounds[col]["min"], self.bounds[col]["max"])
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.df_transformed[col] = pd.Series(
                        self.encoders_priv[col]
                        .fit_transform(self.df_transformed[[col]])
                        .squeeze(),
                        name=col,
                        index=self.df_transformed.index,
                    )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in self.df_transformed.columns:
                    self.encoders[col] = MinMaxScaler(feature_range=(-self.r, self.r))
                    self.df_transformed[col] = pd.Series(
                        self.encoders[col]
                        .fit_transform(self.df_transformed[[col]])
                        .squeeze(),
                        name=col,
                        index=self.df_transformed.index,
                    )

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Instantiate the SMOTE object if not already loaded.

        :param save_path: the path to save the generator
        :return: *None*
        """

        if self.epsilon is None:
            if self._contains_cont_indep_vars and self._contains_cat_indep_vars:
                self._gen = SMOTENC(**self._params)
            elif self._contains_cont_indep_vars:
                self._gen = SMOTE(**self._params)
            else:
                self._gen = SMOTEN(**self._params)
        else:
            if self._params is not None:
                self._gen = DPSmote(**self._params)
            else:
                self._gen = DPSmote(
                    l_connectivity=self.l_connectivity,
                    nu=self.nu,
                    r=self.r,
                    epsilon=self.epsilon,
                    random_state=self._random_state,
                )

        ustandard.save_pickle(
            obj=self._gen, folderpath=save_path, filename=SmoteGenerator.name, date=True
        )

    def display(self) -> None:
        """
        Print the SMOTE parameters.

        :return: *None*
        """

        if self.epsilon is None:
            print("Oversampling generator: ", type(self._gen))
            print("Parameters: ", self._params)
        else:
            print("Parameters: ", self._gen.get_params())

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the oversampling SMOTE method.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        if self.epsilon is None:
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
        else:
            # Prepare the X and y depending on the prediction type (classification or regression)
            if self._prediction_type == "Classification":
                X = self.df_transformed
                y = self._df[self._metadata["variable_to_predict"]]
            else:
                # SMOTE is not defined for regression: we add a fake label 0
                X = self.df_transformed
                y = np.array([0] * len(self.df_transformed))

            # Number of samples in each class
            num_real_samples = len(self._df)
            sampling_strategy = {}
            if self._prediction_type == "Regression":
                sampling_strategy[0] = num_samples
            else:  # Classification
                # The ratio of each class must be preserved
                counts = y.value_counts() / num_real_samples
                counts = (counts * num_samples).round().astype(int)

                if counts.sum() != num_samples:
                    # we add or remove the extra samples due to the rounding to the most frequent class
                    counts.iloc[0] = counts.iloc[0] + (num_samples - counts.sum())
                sampling_strategy = counts.to_dict()

            # Update SMOTE parameters
            self._gen.set_params(sampling_strategy=sampling_strategy)

            # Fit and resample
            df_synth = self._gen.fit_resample(X, y)

            # Post-process the synthetic data
            if self._prediction_type == "Regression":
                # Remove the fake target
                samples = df_synth.drop(columns=["Target"])

                # Convert the range of each feature to the original one
                df_inverse = samples.copy()

                for col in df_inverse.columns:
                    df_inverse[col] = pd.Series(
                        self.encoders[col]
                        .inverse_transform(df_inverse[[col]])
                        .squeeze(),
                        name=col,
                        index=samples.index,
                    )

            else:  # Classification
                # Convert the range of each feature to the original one
                df_inverse = df_synth.copy()

                for col in df_inverse.columns.drop("Target"):
                    df_inverse[col] = pd.Series(
                        self.encoders[col]
                        .inverse_transform(df_inverse[[col]])
                        .squeeze(),
                        name=col,
                        index=df_synth.index,
                    )

                    df_inverse.rename(
                        columns={"Target": self._metadata["variable_to_predict"]},
                        inplace=True,
                    )

            # Align the precision
            for col in self._metadata["continuous"]:
                precision = (
                    self._df[col]
                    .apply(
                        lambda x: len(str(x).split(".")[-1])
                        if isinstance(x, float)
                        else 0
                    )
                    .max()
                )
                df_inverse[col] = df_inverse[col].apply(
                    lambda x: round(x, precision) if isinstance(x, float) else x
                )

                if self._df[col].dtype == "int":
                    df_inverse[col] = df_inverse[col].astype(int)

            df_inverse.to_csv(
                Path(save_path)
                / f"{ustandard.get_date()}_{SmoteGenerator.name}_{num_samples}samples.csv",
                index=False,
            )

            return df_inverse
