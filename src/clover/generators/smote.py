from typing import Dict, Union  # standard library
from pathlib import Path
import warnings

from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN  # 3rd party packages
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from diffprivlib.utils import PrivacyLeakWarning
import pandas as pd
import numpy as np
import torch
from torch import nn

from clover.generators.base import Generator  # local
from clover.generators.models.dpsmote import DPSmote
from clover.utils.postprocessing import transform_data
import clover.utils.standard as ustandard


class SmoteGenerator(Generator):
    """
    Wrapper of the SMOTE (Synthetic Minority Oversampling Technique) Python implementation from imbalanced-learn
    and a differentially private version of SMOTE.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param epsilon: the privacy budget of the differential privacy.
        If epsilon is set to None, a non-DP model will be trained
    :param k_neighbors: the number of neighbors used to find the avatar
    :param nu: the granularity parameter of the uniform grid that the data will be partitioned into
        (applicable to DP generator)
    :param cat_emb_dim: dimension of categorical embeddings (applicable to DP generator)
    :param r: the range each feature will fall into after preprocessing, i.e., [-r, r] (applicable to DP generator)
    :param preprocess_metadata: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening. For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        If not specified, they will be estimated from the real data and a warning will be raised
        (applicable to DP generator)
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
        nu: float = 0.25,
        cat_emb_dim: int = 2,
        r: float = 1,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        bounds = preprocess_metadata

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
            if k_neighbors is not None:
                self._k_neighbors = k_neighbors
            else:
                self._k_neighbors = 5
            self.nu = nu
            self.cat_emb_dim = cat_emb_dim
            self.r = r

            self._df = self._df.copy()
            self.df_original = df.copy()

            # Determine categorical and numerical attributes
            self.num_attrs = metadata["continuous"]
            self.cat_attrs = [
                ele
                for ele in metadata["categorical"]
                if ele != self._metadata["variable_to_predict"]
            ]  # Remove target from features if the task is classification

            # Init bounds dict
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
        # DP-SMOTE:
        # 1. Transform categorical features into continuous features
        # 2. Rescale features so that each feature contains entries in the same range, i.e., [-r, r]
        else:
            if len(self.cat_attrs) > 0:
                # Process categorical variables
                self._df[self.cat_attrs] = self._df[self.cat_attrs].astype("category")
                # Create a dictionary with all the unique categories (all columns in one dict)
                # This will then be used to map back the created names to the original names of the categories
                self._cat_dict = {}

                for cat_attr in self.cat_attrs:
                    # Add unique values and their mapping to the mapping dictionary created above
                    unique_values = self._df[cat_attr].unique()
                    unique_dict = {
                        f"{cat_attr}_{value}": value for value in unique_values
                    }
                    self._cat_dict.update(unique_dict)
                    # add col name to every categorical entry to make them distinguishable for embedding
                    self._df[cat_attr] = (
                        cat_attr + "_" + self._df[cat_attr].astype("str")
                    )

                # Reorder cols
                self._df = self._df[[*self.cat_attrs, *self.num_attrs]]

                # Transform the categorical attributes into ordinal numbers
                vocabulary_classes = []

                for col, series in self._df[self.cat_attrs].items():
                    if col not in self.bounds:
                        warnings.warn(
                            f"List of categories not specified for column '{col}', categories will be extracted from real data for this variable",
                            PrivacyLeakWarning,
                        )
                        unique_value = list(np.unique(self._df[col]))
                    else:
                        unique_value_raw = self.bounds[col]["categories"]
                        unique_value = [
                            col + "_" + str(val) for val in unique_value_raw
                        ]

                    vocabulary_classes += unique_value

                vocabulary_classes = np.sort(vocabulary_classes)
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(vocabulary_classes)
                train_cat_scaled = self._df[self.cat_attrs].apply(
                    self.label_encoder.transform
                )

                # collect unique values of each categorical attribute
                self.vocab_per_attr = {
                    cat_attr: set(train_cat_scaled[cat_attr])
                    for cat_attr in self.cat_attrs
                }

                # Convert to tensor
                train_cat_torch = torch.LongTensor(train_cat_scaled.values)

                # determine number unique categorical tokens
                n_cat_tokens = len(vocabulary_classes)

                self.embedding = nn.Embedding(
                    n_cat_tokens,
                    self.cat_emb_dim,
                    max_norm=None,
                    scale_grad_by_freq=False,
                )  # each value is converted to an n_cat_emb dimension vector
                self.embedding.weight.requires_grad = False

                x_cat_emb = self.embedding(train_cat_torch)  # tensor
                x_cat_emb = x_cat_emb.view(
                    -1, x_cat_emb.shape[1] * x_cat_emb.shape[2]
                ).numpy()

                self.x_cat_emb_col = [
                    f"{col}_{i}"
                    for col in self.cat_attrs
                    for i in range(self.cat_emb_dim)
                ]

                self.df_transformed_cat = pd.DataFrame(
                    x_cat_emb, columns=self.x_cat_emb_col
                )

            self.encoders_priv = {}

            # Encoder to scale the range of each variable to provided bounds and no decoding should be performed
            self.df_transformed_num = self._df[
                self.num_attrs
            ]  # Initiate the transformed continuous data

            # Rescale each variable to the provided bounds to prevent privacy leakage in decoding stage
            for col, series in self.df_transformed_num.items():
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
                    self.df_transformed_num[col] = pd.Series(
                        self.encoders_priv[col]
                        .fit_transform(self.df_transformed_num[[col]])
                        .squeeze(),
                        name=col,
                        index=self.df_transformed_num.index,
                    )

            # Combine the transformed categorical and numerical data
            if len(self.cat_attrs) > 0:
                self.df_transformed = pd.concat(
                    [self.df_transformed_num, self.df_transformed_cat], axis=1
                )
            else:
                self.df_transformed = self.df_transformed_num

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scaler = MinMaxScaler(feature_range=(-self.r, self.r))
                self.df_transformed = pd.DataFrame(
                    self.scaler.fit_transform(self.df_transformed),
                    columns=self.df_transformed.columns,
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
                    k_neighbors=self._k_neighbors,
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

                cat_indep_vars = [
                    i
                    for i, col in enumerate(X.columns)
                    if col in self._metadata["categorical"]
                    and col
                    != self._metadata["variable_to_predict"]  # dependent variable
                ]

                if self._contains_cont_indep_vars and self._contains_cat_indep_vars:
                    self._params["categorical_features"] = cat_indep_vars

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

            # Post-processing
            samples = transform_data(
                df_ref=self._df,
                df_to_trans=samples,
                cont_col=self._metadata["continuous"],
            )

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
                y = self.df_original[self._metadata["variable_to_predict"]]

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
            target = df_synth["Target"]  # Separate target from features for decoding
            samples = df_synth.drop(columns=["Target"])

            # Convert the range of each feature to the original one
            df_inverse = samples.copy()

            df_inverse = pd.DataFrame(
                self.scaler.inverse_transform(df_inverse),
                columns=df_inverse.columns,
            )

            # Separate continuous and categorical features
            df_inverse_num = df_inverse[self.num_attrs]

            if len(self.cat_attrs) > 0:
                df_inverse_cat = df_inverse[self.x_cat_emb_col]

                # Decode categorical features
                embedding_lookup = (
                    self.embedding.weight.data
                )  # get embedding lookup matrix

                # reshape back to n * n_dim_cat * cat_emb_dim
                sample_cat = df_inverse_cat.values.reshape(
                    -1, len(self.cat_attrs), self.cat_emb_dim
                )

                # compute pairwise distances; shape = (# of sample, # of value in lookup, # of attributes)
                distances = torch.cdist(
                    x1=embedding_lookup, x2=torch.Tensor(sample_cat)
                )

                # get the closest distance based on the embeddings that belong to a column category
                z_cat_df = pd.DataFrame(
                    index=range(len(sample_cat)), columns=self.cat_attrs
                )
                nearest_dist_df = pd.DataFrame(
                    index=range(len(sample_cat)), columns=self.cat_attrs
                )
                for attr_idx, attr_name in enumerate(self.cat_attrs):
                    attr_emb_idx = list(
                        self.vocab_per_attr[attr_name]
                    )  # in ascending order
                    attr_distances = distances[:, attr_emb_idx, attr_idx]
                    nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
                    nearest_idx = nearest_idx.numpy()

                    z_cat_df[attr_name] = np.array(attr_emb_idx)[
                        nearest_idx
                    ]  # need to map emb indices back to column indices
                    nearest_dist_df[attr_name] = nearest_values.numpy()

                z_cat_df = z_cat_df.apply(self.label_encoder.inverse_transform)

                df_final = pd.concat([z_cat_df, df_inverse_num], axis=1)

            else:
                df_final = df_inverse_num

            for cat_attr in self.cat_attrs:
                df_final[cat_attr] = df_final[cat_attr].replace(self._cat_dict)

            if self._prediction_type == "Classification":
                df_final[self._metadata["variable_to_predict"]] = target

            # Post-processing
            df_final = transform_data(
                df_ref=self.df_original,
                df_to_trans=df_final,
                cont_col=self.num_attrs,
            )

            df_final.to_csv(
                Path(save_path)
                / f"{ustandard.get_date()}_{SmoteGenerator.name}_{num_samples}samples.csv",
                index=False,
            )

            return df_final
