# Standard library
from typing import Union, List
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local
from .external.synthpop.synthpop import Synthpop
from .base import Generator
import utils.standard as ustandard


class SynthpopGenerator(Generator):
    """
    Wrapper of the Synthpop Python implementation https://github.com/hazy/synthpop.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param variable_order: the order of the variable to construct the sequential trees
    """

    name = "Synthpop"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        variables_order: List[str] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        if generator_filepath is None:
            self._gen = Synthpop(visit_sequence=variables_order, seed=random_state)
        self._df = self._df.copy()
        self._dtypes = None
        self._original_dtypes = df.dtypes.to_dict()

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        self._df[self._metadata["categorical"]] = self._df[
            self._metadata["categorical"]
        ].astype(
            "category"
        )  # Synthpop requires "category" for categories and not object or str

        self._dtypes = self._df.dtypes.apply(
            lambda x: x.name.split("64")[0]
        ).to_dict()  # only 'int' or 'float' supported without any number after

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Construct the sequential trees.

        :param save_path: the path to save the generator
        :return: *None*
        """

        # Deactivate the package prints while fitting the trees
        with ustandard.HiddenPrints():
            self._gen.fit(self._df, self._dtypes)

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=SynthpopGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the constructed sequential trees.

        :return: *None*
        """

        variable_order = list(self._gen.visit_sequence.sort_values().index)

        print("Constructed sequential trees:")
        for i, col in enumerate(variable_order):
            print(f"   {col} has parents {variable_order[:i]}")

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the sequential trees trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        with ustandard.HiddenPrints():  # turn off the prints
            samples = self._gen.generate(num_samples).astype(self._original_dtypes)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{SynthpopGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
