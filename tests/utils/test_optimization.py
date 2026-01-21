import pytest  # 3rd party packages
from pymoo.core.problem import ElementwiseProblem
import numpy as np

import clover.utils.optimization as uoptimization  # local


class TravelingSalesmanProblem(ElementwiseProblem):
    """
    A two-dimensional Traveling Salesman Problem (TSP).

    :param cities: the cities with 2d coordinates provided by a numpy array where each city is represented by a tuple.
    """

    def __init__(self, cities: np.ndarray, **kwargs):
        num_cities = len(cities)
        self._cities = cities

        super().__init__(
            n_var=num_cities, n_obj=1, xl=0, xu=num_cities - 1, vtype=int, **kwargs
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        The function called to compute the fitness function.

        :param x: the solution to evaluate
        :param out: the output dictionary containing the cost
        :param args: for compatibility purposes only
        :param kwargs: for compatibility purposes only
        :return: *None*
        """

        out["F"] = self._get_route_length(x)

    def _get_route_length(self, x: np.ndarray) -> float:
        """
        The cost or fitness function computed as the sum of the Euclidean distances between the cities.

        :param x: the solution to evaluate as a numpy array of indices
        :return: the cost
        """

        x_back_to_initial = np.append(x, x[0])
        city_coord = self._cities[x_back_to_initial]

        line_x = city_coord[:, 0]
        line_y = city_coord[:, 1]

        dist = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))

        return dist


@pytest.fixture(scope="module")
def tsp_problem() -> TravelingSalesmanProblem:
    """
    Initialize the Traveling Salesman Problem with a list of cities.

    :return: the Traveling Salesman Problem object
    """

    cities = np.array(
        [(45, 17), (33, 25), (12, 31), (0, 60), (16, 65), (93, 80), (57, 54)]
    )

    return TravelingSalesmanProblem(cities)


def test_pso(tsp_problem: TravelingSalesmanProblem) -> None:
    """
    Check if the Discrete Particle Swarm Optimization finds the best route.

    :param tsp_problem: the Traveling Salesman Problem object fixture
    :return: *None*
    """

    best_sequence, best_cost = uoptimization.discrete_particle_swarm_optimization(
        problem=tsp_problem
    )

    assert round(best_cost) == 246  # corresponds to the cities in the initialized order
