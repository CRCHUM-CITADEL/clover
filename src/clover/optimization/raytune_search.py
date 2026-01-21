# Standard library
from typing import Type, Callable, Union

# 3rd party packages
import pandas as pd
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from optuna import samplers

# Local
from clover.optimization.base import HyperparametersSearch
from clover.generators import Generator
import config


class RayTuneSearch(HyperparametersSearch):
    """
    Use Ray Tune and Optuna to find the best hyperparameters for the generator. Ray Tune allows parallelization.

    Several optimization algorithms are available, especially Grid Search, Random Search and Bayesian Search.
    "[Bayesian optimization] is typically suited for optimization of high cost functions,
    situations where the balance between exploration and exploitation is important.
    Bayesian optimization works by constructing a posterior distribution of functions [...] that
    best describes the function you want to optimize."

    To learn more:
    Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J. E., & Stoica, I. (2018).
    Tune: A research platform for distributed model selection and training.
    arXiv preprint arXiv:1807.05118.
    https://docs.ray.io/en/latest/tune/index.html

    :cvar name: the name of the hyperparameters search
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param hyperparams: a dictionary with the parameters to optimize and their distribution or
        a function returning the dictionary
    :param generator: the generator class to optimize
    :param objective_function: the cost function
    :param random_state: for reproducibility purposes
    :param use_gpu: flag to use GPU computation power if available to accelerate the learning
    :param resources: a dictionary used to request GPU and CPU resources for each trial
    :param sampler: the algorithm suggesting hyperparameters values. Default to TPESampler.
    :param direction: the direction of optimization (**min** or **max**)
    :param num_iter: the number of steps of bayesian optimization to perform, including the number of startup trials
    :param verbose: 0 (silent), 1 (default), 2 (verbose)
    """

    name = "Ray Tune"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        hyperparams: Union[dict, Callable],
        generator: Type[Generator],
        objective_function: Callable,
        cv_num_folds: int = 1,
        random_state: int = None,
        use_gpu: bool = False,
        resources: dict = None,
        sampler: samplers.BaseSampler = None,
        direction: str = "min",
        num_iter: int = 10,
        verbose: int = 0,
    ):
        super().__init__(
            df,
            metadata,
            hyperparams,
            generator,
            objective_function,
            cv_num_folds,
            random_state,
            use_gpu,
        )

        trainable = lambda config: self._trainable(config)
        if resources is not None:
            trainable = tune.with_resources(
                trainable,
                resources,
            )
        output_path = config.OUTPUT_PATH / "ray"
        output_path.mkdir(parents=True, exist_ok=True)

        self._tuner = tune.Tuner(
            trainable,
            param_space=self._hyperparams,
            tune_config=tune.TuneConfig(
                metric="score",
                mode=direction,
                search_alg=OptunaSearch(
                    sampler=sampler,
                ),
                num_samples=num_iter,
            ),
            run_config=air.config.RunConfig(
                storage_path=str(output_path), verbose=verbose
            ),
        )

    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """

        results = self._tuner.fit()
        best_result = results.get_best_result(metric="score", mode="min")

        self._best_params = best_result.config
        self._best_cost = best_result.metrics["score"]

    def _trainable(self, config: dict) -> None:
        """
        Ray Tune-specific trainable function.

        :param config: the hyperparameters suggested values
        :return: *None*
        """

        callback = lambda cost, _: session.report({"score": cost})
        cost = self._fit(config, callback=callback)

        session.report({"score": cost})

    @property
    def best_params(self):
        return self._best_params

    @property
    def tuner(self):
        return self._tuner
