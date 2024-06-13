# Standard library
from typing import Union, Type
from pathlib import Path

# 3rd party packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

# Local
from generators.base import Generator
from generators.external.private_pgm.mechanisms import mst
from generators.external.private_pgm.mbi.dataset import Dataset
from generators.external.private_pgm.mbi.domain import Domain
import utils.standard as ustandard


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
    :param epsilon: the privacy budget of the differential privacy
    :param delta: the failure probability of the differential privacy
    """

    name = "MST"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        epsilon: float = 1.0,
        delta: float = 1e-9,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._dataset = None

        # Privacy parameters
        self._epsilon = epsilon
        self._delta = delta

        # Encoding
        self._encoder = None

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # The continuous columns must be converted into categorical ones
        kbins = KBinsDiscretizer(n_bins=100, encode="ordinal", strategy="uniform")
        kbins.fit(self._df[self._metadata["continuous"]])
        df_cont = pd.DataFrame(
            kbins.transform(self._df[self._metadata["continuous"]]),
            columns=self._metadata["continuous"],
        ).astype(int)

        # Encode the categorical columns
        self._encoder = (
            OrdinalEncoder()
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
        print("* epsilon", self._epsilon)
        print("* delta", self._delta)

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
        samples[self._metadata["continuous"]] = samples[
            self._metadata["continuous"]
        ].apply(
            lambda x: x + np.round(np.random.rand(len(x)) * 0.5, decimals=2)
        )  # back to "real" floats if the variable type was not ordinal
        samples = samples.astype(self._df.dtypes.to_dict())

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
