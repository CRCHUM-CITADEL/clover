# Standard library
from typing import Type, Callable

# 3rd party packages
import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem

# Local
from clover.optimization.base import HyperparametersSearch
from clover.generators import Generator
import clover.utils.optimization as uoptimization


class DiscreteParticleSwarmOptimizationSearch(HyperparametersSearch):
    """
    Uses a discrete swarm of particles to search the hyperparameters space.

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
    :param direction: the direction of optimization (**min** or **max**)
    :param population_size: the swarm population size
    :param num_iter: the number of iterations to run the search
    """

    name = "Discrete Particle Swarm Optimization Search"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        hyperparams: dict,
        generator: Type[Generator],
        objective_function: Callable,
        cv_num_folds: int = 1,
        random_state: int = None,
        use_gpu: bool = False,
        direction: str = "min",
        population_size: int = 50,
        num_iter: int = 100,
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

        self._direction = direction
        self._population_size = population_size
        self._num_iter = num_iter

        assert (
            len(hyperparams) == 1
        ), "This optimization only works with a unique sequence."
        self._sequence_name = list(self._hyperparams.keys())[0]
        self._default_sequence = self._hyperparams[self._sequence_name]

    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """

        def objective_function(sequence: list):
            cost = self._fit(params={self._sequence_name: sequence})
            if self._direction == "max":
                cost *= -1
            return cost

        problem = SequenceOrderingProblem(
            default_sequence=self._default_sequence,
            objective=objective_function,
        )

        (
            best_sequence,
            self._best_cost,
        ) = uoptimization.discrete_particle_swarm_optimization(
            problem=problem,
            population_size=self._population_size,
            num_epochs=self._num_iter,
        )

        self._best_params = {
            self._sequence_name: list(np.array(self._default_sequence)[best_sequence])
        }
        if self._direction == "max":
            self._best_cost *= -1


class SequenceOrderingProblem(ElementwiseProblem):
    """
    A sequence ordering problem.

    :param default_sequence: the default sequence to optimize
    :param sequence_name: the name of the sequence hyperparameter
    :param objective: the cost function, also fitting a generator and generating samples
    :param kwargs: for compatibility purposes only
    """

    def __init__(self, default_sequence: list, objective: Callable, **kwargs):
        super().__init__(
            n_var=len(default_sequence),
            n_obj=1,
            xl=0,
            xu=len(default_sequence) - 1,
            vtype=int,
            **kwargs,
        )
        self._default_sequence = default_sequence
        self._objective = objective

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        The function called to compute the fitness function.

        :param x: the solution to evaluate
        :param out: the output dictionary containing the cost
        :param args: for compatibility purposes only
        :param kwargs: for compatibility purposes only
        :return: *None*
        """

        sequence = list(np.array(self._default_sequence)[x])

        out["F"] = self._objective(sequence)
