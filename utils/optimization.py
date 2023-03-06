from typing import Tuple  # standard library

from pymoo.algorithms.soo.nonconvex.pso import PSO  # 3rd party packages
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.termination import get_termination
import numpy as np
import pandas as pd


class UniqueIntegerRepair(Repair):
    """
    Ensure that the algorithm is only searching in the feasible space, composed of sequences of unique integers.
    """

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Modify each solution to ensure that each sequence is composed of unique integers.

        :param problem: the problem object defining what to optimize and the objective function
        :param X: the solution for each particle in the swarm
        :param kwargs: for compatibility purposes only
        :return: the solutions in the feasible space
        """
        X_repaired = []
        for x in X:
            # Convert to int and ensure the solution is inside the boundaries
            current_solution = np.clip(np.rint(x).astype(int), problem.xl, problem.xu)

            # Init
            category_set = set(range(len(current_solution)))
            unique_category = np.unique(current_solution)

            # Keep the category if it does not exist in the solution or change it to a missing one
            solution_repaired = np.where(
                pd.Series(current_solution).duplicated(keep="first"),
                -1,
                current_solution,
            )
            solution_repaired[solution_repaired == -1] = list(
                category_set - set(unique_category)
            )

            X_repaired.append(solution_repaired.astype(int))

        return np.array(X_repaired)


def discrete_particle_swarm_optimization(
    problem: Problem, population_size=50, num_epochs: int = 100
) -> Tuple[np.ndarray, float]:
    """
    This optimization algorithm uses a swarm of particles to search the space.
    The objective function does not need to be differentiable.
    The solution is a sequence of unique integers.

    :param problem: the problem object defining what to optimize and the objective function
    :param population_size: the size of the swarm
    :param num_epochs: the number of iterations
    :return: the best solution and its cost
    """

    algorithm = PSO(
        pop_size=population_size,
        sampling=PermutationRandomSampling(),
        repair=UniqueIntegerRepair(),
    )

    termination = get_termination("n_gen", num_epochs)
    res = minimize(problem, algorithm, termination, verbose=False)

    best_solution = res.X
    best_cost = res.F[0]

    return best_solution, best_cost
