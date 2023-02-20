from typing import Union, List  # standard library
from pathlib import Path
import tempfile

from synthpop import Synthpop  # 3rd party packages
import pandas as pd
import numpy as np

from generators.base import Generator  # local
import utils.standard as ustandard
import utils.learning as ulearning
from metrics.utility.population import Distinguishability


class SynthpopGenerator(Generator):
    """
    Wrapper of the Synthpop Python implementation https://github.com/hazy/synthpop.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param generator_filepath: the path of the generator to sample from if it exists
    :param variable_order: the order of the variable to construct the sequential trees
    """

    name = "Synthpop"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        generator_filepath: Union[Path, str] = None,
        variables_order: List[str] = None,
    ):
        super().__init__(df, metadata, generator_filepath)

        self._gen = (
            Synthpop(visit_sequence=variables_order)
            if generator_filepath is None
            else ustandard.load_pickle(filepath=generator_filepath)
        )
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
        self._gen.fit(self._df, self._dtypes)

        ustandard.save_pickle(
            obj=self._gen, path=save_path, filename=SynthpopGenerator.name
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

        samples = self._gen.generate(num_samples).astype(self._original_dtypes)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{SynthpopGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples

    def search_hyperparameters(self, **kwargs) -> dict:
        """
        Use randomization to find the best order of the variables to train the sequential trees.
        The hinge loss applied to the distinguishability metric is used as objective function.

        :param kwargs: a dict containing the number of iterations **num_iter**
        :return: a dictionary with the **variables_order** as key
        """

        assert {"num_iter"} <= kwargs.keys()

        columns = list(self._df.columns)

        # Init
        iter = 0
        best_sequence = columns
        best_cost = self._objective_function(sequence=best_sequence)

        while best_cost > 0 and iter < kwargs["num_iter"]:
            sequence = list(np.random.choice(columns, size=len(columns), replace=False))
            cost = self._objective_function(sequence=sequence)
            print(sequence, cost)
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence
            iter += 1

        res = {"variables_order": best_sequence, "cost": best_cost}

        return res

    def _objective_function(self, sequence: List[str]):
        # Synthetize the data with the given order of the variables
        gen = SynthpopGenerator(
            df=self._df, metadata=self._metadata, variables_order=sequence
        )
        gen.preprocess()
        with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated samples
            gen.fit(save_path=temp_dir)
            samples = gen.sample(save_path=temp_dir, num_samples=len(self._df))

        # Compute the distinguishability metric
        dist = Distinguishability(num_repeat=20, num_folds=0)
        propensity_score = dist.compute(
            df_real=self._df, df_synthetic=samples, metadata=self._metadata
        )["average"]["propensity_mse"]

        # Compute the hinge loss based on the distinguishability score
        loss = ulearning.hinge_loss(propensity_score, threshold=0.05)

        return loss
