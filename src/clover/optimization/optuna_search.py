# Standard library
from typing import Type, Callable, Union

# 3rd party packages
import pandas as pd
import optuna
from optuna import samplers, pruners

# Local
from clover.optimization.base import HyperparametersSearch
from clover.generators import Generator


class OptunaSearch(HyperparametersSearch):
    """
    Use Optuna to find the best hyperparameters for the generator.

    Several optimization algorithms are available, especially Grid Search, Random Search and Bayesian Search.
    "[Bayesian optimization] is typically suited for optimization of high cost functions,
    situations where the balance between exploration and exploitation is important.
    Bayesian optimization works by constructing a posterior distribution of functions [...] that
    best describes the function you want to optimize."

    To learn more:
    Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
    Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
    https://optuna.org/

    :cvar name: the name of the hyperparameters search
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param hyperparams: a dictionary with the parameters to optimize and their distribution or
        a function returning the dictionary
    :param generator: the generator class to optimize
    :param objective_function: the cost function
    :param cv_num_folds: the number of folds for the cross-validation.
        Set to 0 or 1 for deactivating the cross-validation.
    :param random_state: for reproducibility purposes
    :param use_gpu: flag to use GPU computation power if available to accelerate the learning
    :param sampler: the algorithm suggesting hyperparameters values. Default to TPESampler.
    :param pruner: the algorithm early stopping the unsuccessful trials. No pruning by default.
    :param direction: the direction of optimization (**minimize** or **maximize**)
    :param num_iter: the number of steps of bayesian optimization to perform, including the number of startup trials
    :param verbose: whether to print the INFO logs (1) or not (0)
    """

    name = "Optuna"

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
        sampler: samplers.BaseSampler = None,
        pruner: pruners.BasePruner = pruners.NopPruner,
        direction: str = "minimize",
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

        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Block for coherence with other optimizers, which use "min" or "max" instead of "minimize" and "maximize".
        if direction == "max" or direction == "maximize":
            direction = "maximize"
        else:
            direction = "minimize"

        self._study = optuna.create_study(
            direction=direction, sampler=sampler, pruner=pruner
        )
        self._num_iter = num_iter

    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """

        # Optuna-specific objective function
        def objective(trial):
            params_to_explore = self._hyperparams(trial)
            callback = lambda cost, step: trial.report(cost, step=step)
            cost = self._fit(params_to_explore, callback=callback)
            return cost

        self._study.optimize(func=objective, n_trials=self._num_iter)

        self._best_params = self._study.best_trial.params
        self._best_cost = self._study.best_trial.value
