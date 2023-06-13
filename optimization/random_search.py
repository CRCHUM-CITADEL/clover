# Standard library
from typing import Type, Callable

# 3rd party packages
import pandas as pd
import numpy as np

# Local
from optimization.base import HyperparametersSearch
from generators.base import Generator


class RandomSearch(HyperparametersSearch):
    """
    Random hyperparameters search.

    :cvar name: the name of the hyperparameters search
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param hyperparams: a dictionary with the parameters to optimize and their distribution
    :param hyperparams_type: a dictionary with the parameters to optimize and
        their type (should be int, float or sequence)
    :param generator: the generator class to optimize
    :param objective_function: the cost function (must be positive and 0 the target value)
    :param random_state: for reproducibility purposes
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    :param num_iter: the number of iterations to run the search
    """

    name = "Random Search"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        hyperparams: dict,
        hyperparams_type: dict,
        generator: Type[Generator],
        objective_function: Callable,
        random_state: int = None,
        use_gpu: bool = False,
        num_iter: int = 100,
    ):
        super().__init__(
            df,
            metadata,
            hyperparams,
            hyperparams_type,
            generator,
            objective_function,
            random_state,
            use_gpu,
        )
        self._num_iter = num_iter

    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """

        # Init
        iter = 0
        self._best_params = self._get_random_hyperparameters()
        self._best_estimator, df_synth = self._fit_generator(params=self._best_params)
        self._best_cost = self._objective_function(
            df=self._df,
            df_to_compare=df_synth,
            metadata=self._metadata,
            minimize=True,
            use_gpu=self._use_gpu,
        )

        while self._best_cost > 0 and iter < self._num_iter:
            params = self._get_random_hyperparameters()
            estimator, df_synth = self._fit_generator(params=params)
            cost = self._objective_function(
                df=self._df,
                df_to_compare=df_synth,
                metadata=self._metadata,
                minimize=True,
                use_gpu=self._use_gpu,
            )

            self._results[iter] = {  # update results
                "hyperparameters": params,
                "cost": cost,
            }

            if cost < self._best_cost:  # update best if the cost is reduced
                self._best_cost = cost
                self._best_estimator = estimator
                self._best_params = params

            iter += 1

    def _get_random_hyperparameters(self) -> dict:
        """
        Draw randomly a parameter from the user-specified distribution.

        :return: a dictionary containing the hyperparameters randomly picked
        """
        hyperparams = {}

        for key, value in self._hyperparams.items():
            if self._hyperparams_type[key] == "sequence":
                hyperparams[key] = list(
                    np.random.choice(value, size=len(value), replace=False)
                )
            else:  # single value
                hyperparams[key] = np.random.choice(value)

        return hyperparams
