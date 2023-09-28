Discrete Particle Swarm Optimization
=====================================

Basic continuous Particule Swarm Optimization (PSO)
---------------------------------------------------

Particle Swarm Optimization (PSO) is a population-based stochastic optimization algorithm that draws inspiration from the collective behavior observed in bird flocks or fish schools. In PSO, a population of potential solutions, referred to as particles, collaboratively explores the search space in an iterative manner to locate the optimal or near-optimal solution. Each particle represents a candidate solution and navigates through the search space, guided by its own experience and the knowledge shared by its neighboring particles. The objective function is then evaluated for each particle, providing a measure of its fitness or quality at the current position. Each particle maintains a personal best position, reflecting the position that has yielded the best fitness value thus far. If the current position surpasses the personal best, the personal best is updated accordingly. Additionally, the global best position, representing the best solution discovered by any particle in the entire population, is continuously updated based on the personal best positions of all particles. To adapt their movement, particles update their velocities and positions based on their current state, personal best, and global best. The velocity influences the direction and magnitude of the particle's movement, while the new position determines its location in the search space. These steps are iteratively repeated for a specified number of iterations or until a termination criterion is met, such as achieving a desired fitness value or exhausting computational resources.

PSO may encounter challenges such as premature convergence or being trapped in local optima. To mitigate these limitations, researchers have proposed various enhancements and extensions to the algorithm. These include incorporating adaptive parameters, hybridizing PSO with other algorithms, or leveraging problem-specific knowledge to improve performance and convergence. In summary, Particle Swarm Optimization is a powerful and widely employed optimization algorithm that harnesses the collective behavior of particles to efficiently explore and discover optimal or near-optimal solutions within a given search space. Its effectiveness, ease of implementation, and ability to handle complex objective functions make it a valuable tool for solving a wide range of optimization problems. Ongoing research continues to advance PSO and its adaptations, paving the way for further improvements in solving complex optimization challenges.

Discrete PSO, applied to the Traveling Salesman Problem
-------------------------------------------------------

The Traveling Salesman Problem (TSP) is a well-known optimization problem that involves finding the shortest route for a salesman to visit a set of cities and return to the starting city, while visiting each city exactly once. Due to its computational complexity, various optimization algorithms have been employed to tackle this challenge. One such algorithm is Particle Swarm Optimization (PSO). When applied to the discrete TSP, several adaptations are introduced to effectively optimize the salesman's route. In the TSP variant of PSO, particles represent potential solutions, with each particle corresponding to a specific permutation of the cities to be visited. The fitness of each particle is evaluated based on the total distance or cost of the corresponding route. The algorithm initializes a population of particles randomly within the search space, assigning them random positions and velocities. Throughout the iterations, particles adjust their velocities and positions, guided by their personal best (the best route found by the particle itself) and the global best (the best route discovered by any particle in the population). The position perturbation operation, replacing the traditional velocity update, modifies the permutation of cities within each particle to explore different route possibilities. By evaluating fitness values and updating positions based on personal and global bests, PSO efficiently explores the solution space, striving to find high-quality solutions for the TSP.

Clover implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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


See the notebook `Tune hyperparameters <../../tutorial/tune_hyperparameters.nblink>`_ to learn how to use the optimizer.

References
----------

- `Luping Fang, Pan Chen, Shihua Liu: Particle Swarm Optimization with Simulated Annealing for TSP, Proceedings of the 6th WSEAS Int. Conf. on Artificial Intelligence, Knowledge Engineering and Data Bases, Corfu Island, Greece, February 16-19, 2007, pp. 206-210 <https://dl.acm.org/doi/10.5555/1348485.1348522>`_

- https://github.com/CRCHUM-CITADEL/clover/blob/main/optimization/discrete_pso_search.py

- https://github.com/CRCHUM-CITADEL/clover/blob/main/utils/optimization.py

- https://github.com/CRCHUM-CITADEL/clover/blob/main/tests/utils/test_optimization.py

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/tune_hyperparameters.ipynb