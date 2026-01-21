# Standard library
from abc import ABCMeta, abstractmethod
from typing import Union, Type, Callable
import inspect
import tempfile

# 3rd party packages
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# Local
from clover.generators import Generator


class HyperparametersSearch(metaclass=ABCMeta):
    """
    Abstract class providing the template to follow for each type of hyperparameters search.

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
        hyperparams: Union[dict, Callable],
        generator: Type[Generator],
        objective_function: Callable,
        cv_num_folds: int = 1,
        random_state: int = None,
        use_gpu: bool = False,
    ):
        self._df = df
        self._metadata = metadata
        self._hyperparams = hyperparams
        self._generator = generator
        self._objective_function = objective_function
        self._cv_num_folds = cv_num_folds
        self._random_state = random_state
        self._use_gpu = use_gpu

        if isinstance(
            hyperparams, dict
        ):  # TODO: see how to check the hyperparameters for optuna
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

    def _fit(self, params: dict, callback: Callable = None) -> float:
        """
        Invoked repeatedly during optimization to fit the generator, generate samples and
        compute the objective function.

        :param params: the hyperparameters to test
        :param callback: the callback to report the intermediate results
        :return: the cost
        """

        def objective(df):
            gen = self._generator(df=df["train"], metadata=self._metadata, **params)
            gen.preprocess()

            df_synth = {}
            with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated samples
                gen.fit(save_path=temp_dir)
                df_synth["train"] = gen.sample(
                    save_path=temp_dir, num_samples=len(df["train"])
                )
                df_synth["test"] = gen.sample(
                    save_path=temp_dir, num_samples=len(df["test"])
                )

            cost = self._objective_function(
                df=df,
                df_to_compare=df_synth,
                metadata=self._metadata,
                use_gpu=self._use_gpu,
            )

            return cost

        if self._cv_num_folds < 2:  # no cross-validation
            return objective({"train": self._df, "test": self._df})

        X = self._df.drop(columns=self._metadata["variable_to_predict"])
        y = self._df[self._metadata["variable_to_predict"]]

        if self._metadata["variable_to_predict"] in self._metadata["categorical"]:
            kf = StratifiedKFold(n_splits=self._cv_num_folds, shuffle=True)
        else:
            kf = KFold(n_splits=self._cv_num_folds, shuffle=True)

        costs = []

        for step, (train_index, test_index) in enumerate(kf.split(X, y)):
            cost = objective(
                {"train": self._df.iloc[train_index], "test": self._df.iloc[test_index]}
            )
            costs.append(cost)

            if callback is not None:
                callback(cost, step)

        loss = np.mean(costs)

        return loss
