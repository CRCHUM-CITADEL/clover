# Standard library
from abc import ABCMeta, abstractmethod
from typing import Tuple, Type, Callable
import inspect
import tempfile

# 3rd party packages
import pandas as pd

# Local
from generators.base import Generator


class HyperparametersSearch(metaclass=ABCMeta):
    """
    Abstract class providing the template to follow for each type of hyperparameters search.

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
    """

    name: str

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        """
        :return: the name of the generator
        """

    @property
    def results(self) -> dict:
        """
        Returns all the tested combinations if the optimizer supports it, else empty.

        :return: a dictionary with the dictionary of tested **hyperparameters** and their respective **cost**
        """
        return self._results

    @property
    def best_estimator(self) -> Generator:
        """
        Returns the best estimator if the optimizer supports it, else *None*.

        :return: the best generator
        """
        return self._best_estimator

    @property
    def best_cost(self) -> float:
        """
        Returns the cost of the best solution.

        :return: the best cost
        """
        return self._best_cost

    @property
    def best_params(self) -> dict:
        """
        Returns the best hyperparameters.

        :return: a dictionary containing the hyperparameters
        """
        return self._best_params

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
    ):
        self._df = df
        self._metadata = metadata
        self._hyperparams = hyperparams
        self._hyperparams_type = hyperparams_type
        self._generator = generator
        self._objective_function = objective_function
        self._random_state = random_state
        self._use_gpu = use_gpu

        self._check_hyperparameters()

        # Parameters available after the search
        self._results = {}
        self._best_estimator = None
        self._best_cost = None
        self._best_params = None

    def _check_hyperparameters(self) -> None:
        """
        Assert that the init parameters are consistent.

        :return: *None*
        """
        assert len(self._hyperparams) != 0, "No parameter to optimize"
        assert set(self._hyperparams.keys()) == set(
            self._hyperparams_type.keys()
        ), "The hyperparameters keys should match the hyperparameters type keys"
        assert set(self._hyperparams_type.values()).issubset(
            ["int", "float", "sequence"]
        ), "The hyperparameters type must be int, float or sequence. Not yet implemented for other types."

        generator_parameters = set(inspect.signature(self._generator).parameters)
        assert set(self._hyperparams.keys()).issubset(
            generator_parameters
        ), f"The parameters to optimize must be parameters of {self._generator.name}"

    @abstractmethod
    def fit(self) -> None:
        """
        Find the best hyperparameters for the generator.

        :return: *None*
        """
        pass

    def _fit_generator(self, params: dict) -> Tuple[Generator, pd.DataFrame]:
        """
        Invoked repeatedly during optimization to fit the generator and generate samples.

        :param params: the hyperparameters to test
        :return: the fitted generator and the generated samples
        """

        gen = self._generator(df=self._df, metadata=self._metadata, **params)
        gen.preprocess()

        with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated samples
            gen.fit(save_path=temp_dir)
            samples = gen.sample(save_path=temp_dir, num_samples=len(self._df))

        return gen, samples
