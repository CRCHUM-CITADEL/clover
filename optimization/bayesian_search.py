# Standard library
from typing import Type, Callable

# 3rd party packages
import pandas as pd
from bayes_opt import BayesianOptimization

# Local
from optimization.base import HyperparametersSearch
from generators.base import Generator


class BayesianSearch(HyperparametersSearch):
    """
    Use Bayesian optimization to find the best hyperparameters for the generator.

    "[Bayesian optimization] is typically suited for optimization of high cost functions,
    situations where the balance between exploration and exploitation is important.
    Bayesian optimization works by constructing a posterior distribution of functions [...] that
    best describes the function you want to optimize."

    To learn more:
    Fernando Nogueira (2014)
    Bayesian optimization: Open source constrained global optimization tool for Python
    https://github.com/fmfn/BayesianOptimization

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
    :param init_points: the number of steps of random exploration (helps diversify the exploration space).
    :param num_iter: the number of steps of bayesian optimization to perform. This starts after the exploration.
    :param verbose: whether to print the parameters to explore
    """

    name = "Bayesian Optimization"

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
        init_points: int = 5,
        num_iter: int = 30,
        verbose: bool = True,
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

        self._init_points = init_points
        self._num_iter = num_iter
        self._verbose = verbose

    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """

        # Distance function
        def dist_function(**kwargs):
            params_to_explore = self._convert_parameters(kwargs)

            _, df_synth = self._fit_generator(params_to_explore)
            cost = self._objective_function(
                df=self._df,
                df_to_compare=df_synth,
                metadata=self._metadata,
                minimize=False,
                use_gpu=self._use_gpu,
            )
            return cost

        # The Bayesian Optimization object is created, and the optimization performed.
        ctgan_bo = BayesianOptimization(
            dist_function, self._hyperparams, random_state=self._random_state
        )
        ctgan_bo.maximize(init_points=self._init_points, n_iter=self._num_iter)

        # The best parameters are converted to the actual values (instead of the floats).
        self._best_params = self._convert_parameters(ctgan_bo.max["params"])
        self._best_cost = ctgan_bo.max["target"]

    def _convert_parameters(self, params: dict) -> dict:
        """
        Cast a dictionary of parameters to their specified type

        :param params: a dictionary of the parameters to optimize
        :return: a dictionary containing the hyperparameters cast to their initial type
        """

        hyperparams = {}

        for key in params.keys():
            hyperparams[key] = params[key].astype(self._hyperparams_type[key])

        return hyperparams
