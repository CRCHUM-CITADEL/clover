# Standard library
from typing import Dict, Union, List
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local
from .external.dpart.dpart.engines import DPSynthpop
from .base import Generator
import utils.standard as ustandard


class DPSynthpopGenerator(Generator):
    """
    Differential private Synthpop Python implementation based on the dpart framework https://github.com/hazy/dpart.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param methods: defines the specific method each column should be modelled by.
        The default methods to model continuous and discrete columns are DP-Linear Regression and DP-Logistic Regression.
    :param epsilon: the privacy budget of the differential privacy
    :param slack: slack allowed in delta spend for training the DP ML models.
        This is to show how to gain higher privacy guarantee in terms of epsilon if a slack is allowed.
    :param bounds: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage is happening.
    :param variable_order: the order in which the joint distribution is broken down into a sequence of conditionals
    :param prediction_matrix: specify the collection of already visited columns to be used as features for each unvisited column.
        It could be set to "infer" to optimize the variable_order by maximizing the information gain.
    :param n_parents: maximum number of columns to be considered as features to predict a target
    """

    name = "DPSynthpop"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        methods: dict = None,
        epsilon: Union[float, Dict[str, Union[float, Dict[str, float]]]] = None,
        slack: float = 0.0,
        bounds: Dict[str, List] = None,
        variables_order: List[str] = None,
        prediction_matrix: Union[str, Dict[str, List[str]]] = None,
        n_parents: int = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        n_col = df.shape[1]
        if n_parents == None:
            self.n_parents = n_col
        else:
            self.n_parents = n_parents

        if generator_filepath is None:
            self._gen = DPSynthpop(
                methods=methods,
                epsilon=epsilon,
                slack=slack,
                bounds=bounds,
                visit_order=variables_order,
                prediction_matrix=prediction_matrix,
                n_parents=self.n_parents,
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
        )  # requires "object", "(byte-)string" or boolean for categories

        self._dtypes = self._df.dtypes.to_dict()

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Fit a model for each target column.

        :param save_path: the path to save the generator
        :return: *None*
        """

        # Deactivate the package prints while fitting the models
        with ustandard.HiddenPrints():
            self._gen.fit(self._df)

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=DPSynthpopGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the visit order.

        :return: *None*
        """

        variable_order = list(self._gen.dep_manager.visit_order)

        print("Variable visit order:")
        for i, col in enumerate(variable_order):
            print(f"   {col} has parents {variable_order[:i]}")

        print("")
        print("Privacy budget spent:")
        print(self._gen.budget_acc.total())

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the models trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        with ustandard.HiddenPrints():  # turn off the prints
            samples = self._gen.sample(num_samples).astype(self._original_dtypes)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{DPSynthpopGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
