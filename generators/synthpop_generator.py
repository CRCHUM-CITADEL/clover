from typing import Union, List  # standard library
from pathlib import Path
import tempfile

from synthpop import Synthpop  # 3rd party packages
from pymoo.core.problem import ElementwiseProblem
import pandas as pd
import numpy as np

from generators.base import Generator  # local
import utils.standard as ustandard
import utils.learning as ulearning
import utils.optimization as uoptimization
from metrics.utility.population import Distinguishability


class SynthpopGenerator(Generator):
    """
    Wrapper of the Synthpop Python implementation https://github.com/hazy/synthpop.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param variable_order: the order of the variable to construct the sequential trees
    """

    name = "Synthpop"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        variables_order: List[str] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._gen = (
            Synthpop(visit_sequence=variables_order, seed=random_state)
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
        with ustandard.HiddenPrints():
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

        with ustandard.HiddenPrints():  # turn off the prints
            samples = self._gen.generate(num_samples).astype(self._original_dtypes)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{SynthpopGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples

    def search_hyperparameters(self, **kwargs) -> dict:
        """
        Use Particule Swarm Optimization (pso) or randomization to find the best order of the variables
        to train the sequential trees.

        :param kwargs: a dict containing the search type **search** ("pso" or "random"), the number of iterations
          **num_iter** and the population size **population_size** for pso.
        :return: a dictionary with the **variables_order** and the **cost** as keys
        """

        assert {"search"} <= kwargs.keys()
        if kwargs["search"] == "pso":
            return self._particule_swarm_optimization_search(**kwargs)
        else:
            return self._random_search(**kwargs)

    def _random_search(self, **kwargs) -> dict:
        """
        Use randomization to find the best order of the variables to train the sequential trees.
        The hinge loss applied to the distinguishability metric is used as objective function.

        :param kwargs: a dict containing the number of iterations **num_iter**
        :return: a dictionary with the **variables_order** and the **cost** as keys
        """

        assert {"num_iter"} <= kwargs.keys()

        # Init
        iter = 0
        num_cols = len(self._df.columns)
        best_sequence = np.arange(num_cols)
        problem = SequenceOrderingProblem(self._df, self._metadata)
        best_cost = problem._objective_function(solution=best_sequence)

        while best_cost > 0 and iter < kwargs["num_iter"]:
            sequence = np.random.choice(
                np.arange(num_cols), size=num_cols, replace=False
            )
            cost = problem._objective_function(solution=sequence)
            print(sequence, cost)
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence
            iter += 1

        best_sequence = list(np.array(self._df.columns)[best_sequence])

        res = {"variables_order": best_sequence, "cost": best_cost}

        return res

    def _particule_swarm_optimization_search(self, **kwargs) -> dict:
        """
        Use Particle Swarm Optimization (pso) to find the best order of the variables to train the sequential trees.
        The hinge loss applied to the distinguishability metric is used as objective function.

        :param kwargs: a dict containing the number of iterations **num_iter**
          and the population size **population_size**
        :return: a dictionary with the **variables_order** and the **cost** as keys
        """

        assert {"num_iter", "population_size"} <= kwargs.keys()

        problem = SequenceOrderingProblem(self._df, self._metadata)
        best_sequence, best_cost = uoptimization.discrete_particle_swarm_optimization(
            problem=problem,
            population_size=kwargs["population_size"],
            num_epochs=kwargs["num_iter"],
        )
        best_sequence = list(np.array(self._df.columns)[best_sequence])

        res = {"variables_order": best_sequence, "cost": best_cost}

        return res


class SequenceOrderingProblem(ElementwiseProblem):
    """
    A sequence ordering problem with the Distinguishability score as objective function.

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param kwargs: for compatibility purposes only
    """

    def __init__(self, df: pd.DataFrame, metadata: dict, **kwargs):
        super().__init__(
            n_var=df.shape[1], n_obj=1, xl=0, xu=df.shape[1] - 1, vtype=int, **kwargs
        )

        self._df = df
        self._metadata = metadata

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        The function called to compute the fitness function.

        :param x: the solution to evaluate
        :param out: the output dictionary containing the cost
        :param args: for compatibility purposes only
        :param kwargs: for compatibility purposes only
        :return: *None*
        """

        out["F"] = self._objective_function(x)

    def _objective_function(self, solution: np.ndarray) -> float:
        """
        The cost or fitness function computed as the Hinge loss applied to the distinguishability metric.

        :param solution: the solution to evaluate as a numpy array of indices
        :return: the cost
        """

        # The solution is a list of indices instead of a list of column names
        sequence = list(np.array(self._df.columns)[solution])

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
