# Standard library
from typing import Dict, Union, Type
from pathlib import Path
import warnings

# 3rd party packages
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, MinMaxScaler
from diffprivlib.utils import PrivacyLeakWarning

# Local
from clover.generators.base import Generator
from clover.generators.external.private_pgm.mechanisms import mst
from clover.generators.external.private_pgm.mbi.dataset import Dataset
from clover.generators.external.private_pgm.mbi.domain import Domain
from clover.utils.postprocessing import transform_data
import clover.utils.standard as ustandard


class MSTGenerator(Generator):
    """
    Wrapper of the Maximum Spanning Tree (MST) method from Private-PGM repo:
    https://github.com/ryan112358/private-pgm/tree/master.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param nbins: number of bins to discretized continuous variables
    :param epsilon: the privacy budget of the differential privacy
    :param delta: the failure probability of the differential privacy
    :param preprocess_metadata: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening. For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        If not specified, they will be estimated from the real data and a warning will be raised
        (for differentially private model)
    """

    name = "MST"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        nbins: int = 100,
        epsilon: float = 1.0,
        delta: float = 1e-9,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        bounds = preprocess_metadata

        self._dataset = None

        # Privacy parameters
        self._epsilon = epsilon
        self._delta = delta

        # Encoding
        self._nbins = nbins
        self._scaler = {}
        self._kbins = None
        self._encoder = None

        # Bounds
        if bounds is None:
            self.bounds = {}
        else:
            self.bounds = bounds

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # Rescale continuous variables to min and max to prevent privacy leakage
        df_cont_rescaled = pd.DataFrame(columns=self._metadata["continuous"])

        for col, series in self._df[self._metadata["continuous"]].items():
            if col not in self.bounds:
                warnings.warn(
                    f"upper and lower bounds not specified for column '{col}'",
                    PrivacyLeakWarning,
                )
                self.bounds[col] = {"min": series.min(), "max": series.max()}
            self._scaler[col] = MinMaxScaler(
                feature_range=(self.bounds[col]["min"], self.bounds[col]["max"])
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_cont_rescaled[col] = pd.Series(
                    self._scaler[col].fit_transform(self._df[[col]]).squeeze(),
                    name=col,
                    index=self._df.index,
                )

        # The continuous columns must be converted into categorical ones
        self._kbins = KBinsDiscretizer(
            n_bins=self._nbins, encode="ordinal", strategy="uniform"
        )
        self._kbins.fit(df_cont_rescaled[self._metadata["continuous"]])
        df_cont = pd.DataFrame(
            self._kbins.transform(df_cont_rescaled[self._metadata["continuous"]]),
            columns=self._metadata["continuous"],
        ).astype(int)

        # Encode the categorical columns
        categories = []
        for col in self._metadata["categorical"]:
            if col in self.bounds:
                if is_numeric_dtype(self._df[col]):
                    categories.append(sorted(self.bounds[col]["categories"]))
                else:
                    categories.append(self.bounds[col]["categories"])
            else:
                if is_numeric_dtype(self._df[col]):
                    categories.append(sorted(list(self._df[col].unique())))
                else:
                    categories.append(list(self._df[col].unique()))

                warnings.warn(
                    f"List of categories not specified for column '{col}', categories will be extracted from real data for this variable",
                    PrivacyLeakWarning,
                )

        self._encoder = OrdinalEncoder(
            categories=categories
        )  # One-Hot encoder does not work with the method
        data = self._encoder.fit_transform(self._df[self._metadata["categorical"]])

        df_cat = pd.DataFrame(
            data,
            columns=self._metadata["categorical"],
        )

        # Merge the preprocessed dataframes
        df = pd.concat([df_cont, df_cat], axis=1)

        # Create the domain metadata
        domain = df.nunique().to_dict()

        # Create the Dataset object for Private-pgm
        self._dataset = Dataset(df, Domain.fromdict(domain))

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Define and save the MST parameters. The fit is executed with the sampling.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._gen = MSTClass(self._dataset, self._epsilon, self._delta)

        ustandard.save_pickle(
            obj=self._gen, folderpath=save_path, filename=MSTGenerator.name, date=True
        )

    def display(self) -> None:
        """
        Print the MST parameters.

        :return: *None*
        """
        print("Generator: Maximum Spanning Tree MST")
        print("Parameters:")
        print("* epsilon", self._gen._epsilon)
        print("* delta", self._gen._delta)

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the MST method.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        samples = self._gen.generate(num_samples)

        # Decode ordinal
        if len(self._metadata["categorical"]) > 0:
            samples[self._metadata["categorical"]] = self._encoder.inverse_transform(
                samples[self._metadata["categorical"]]
            )

        # Transform to origin
        samples = samples[self._df.columns]  # same initial columns order
        samples[self._metadata["continuous"]] = self._kbins.inverse_transform(
            samples[self._metadata["continuous"]]
        )

        for idx, col in enumerate(self._metadata["continuous"]):
            half_bin_width = (
                self._kbins.bin_edges_[idx][1] - self._kbins.bin_edges_[idx][0]
            ) / 2

            samples[col] = samples[col] + np.random.uniform(
                -half_bin_width, half_bin_width, size=len(samples)
            )

        # Post-processing
        samples = transform_data(
            df_ref=self._df,
            df_to_trans=samples,
            cont_col=self._metadata["continuous"],
        )

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{MSTGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples


class MSTClass:
    """
    Class wrapping the Maximum Spanning Tree (MST) method. Only used inside MSTGenerator for compatibility purposes.

    :param dataset: the Dataset object as described by Private-PGM library.
    :param epsilon: the privacy budget of the differential privacy
    :param delta: the failure probability of the differential privacy
    """

    def __init__(self, dataset: Type[Dataset], epsilon: float, delta: float):
        self._dataset = dataset
        self._epsilon = epsilon
        self._delta = delta

    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Fit the MST model and generate synthetic data.

        :param num_samples: the number of samples to generate
        :return: the generated samples
        """
        data = mst.MST(
            data=self._dataset,
            epsilon=self._epsilon,
            delta=self._delta,
            num_samples=num_samples,
        )

        return data.df
