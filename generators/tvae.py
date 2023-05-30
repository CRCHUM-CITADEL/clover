from typing import Union # standard library

import pandas as pd
from pathlib import Path
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from bayes_opt import BayesianOptimization

from generators.base import Generator  # local
import utils.standard as ustandard
from metrics.utility.population import Distinguishability


class TVAEGenerator(Generator):
    """
    Also a wrapper of a data synthesizer available in the SDV package.
    The synthesizer is TVAE, a VAE for tabular data.
    https://github.com/sdv-dev

    See article for more information:
    Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
    Modeling tabular data using conditional GAN.
    Advances in Neural Information Processing Systems, 32.
    https://arxiv.org/abs/1907.00503
    """

    name = "TVAE"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        epochs=300,
        batch_size=100,
        compress_dims=(249, 249),
        decompress_dims=(249, 249),
    ):
        """
        Initialize the class.

        :param df: DataFrame containing the original data to be synthesized.
        :type df: pd.DataFrame
        :param metadata: Metadata describing the variables.
        :type metadata: dict
        :param random_state: Random seed for reproducibility, defaults to None.
        :type random_state: int, optional
        :param generator_filepath: File path to a generator object, defaults to None.
        :type generator_filepath: Union[Path, str], optional
        :param epochs: Number of training epochs, defaults to 300.
        :type epochs: int, optional
        :param batch_size: Batch size for training, defaults to 100.
        :type batch_size: int, optional
        :param compress_dims: Size of the hidden layers in the encoder, defaults to (249, 249).
        :type compress_dims: tuple, optional
        :param decompress_dims: Size of the hidden layers in the decoder, defaults to (249, 249).
        :type decompress_dims: tuple, optional
        """
        super().__init__(df, metadata, random_state, generator_filepath)

        self._params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "compress_dims": compress_dims,
            "decompress_dims": decompress_dims,
        }
        self._tvae_metadata = None

        self.preprocess()

        self._gen = (
            TVAESynthesizer(self._tvae_metadata, **self._params)
            if generator_filepath is None
            else ustandard.load_pickle(filepath=generator_filepath)
        )

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        temp_dict = {}
        final_dict = {}

        for col in self._metadata["continuous"]:
            temp_dict[col] = {"sdtype": "numerical"}
        for col in self._metadata["categorical"]:
            temp_dict[col] = {"sdtype": "categorical"}
        final_dict["columns"] = temp_dict

        self._tvae_metadata = SingleTableMetadata.load_from_dict(final_dict)

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._gen.fit(self._df)

        ustandard.save_pickle(
            obj=self._gen, path=save_path, filename=TVAEGenerator.name
        )

    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        print("TVAE synthesizer parameters: ")
        print(self._gen.get_parameters())

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the synthesizer trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        samples = self._gen.sample(num_rows=num_samples)

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{TVAEGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples

    def search_hyperparameters(
        self, init_points=5, n_iter=30, verbose=True, **kwargs
    ) -> dict:
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

        :param init_points: number of steps of random exploration (helps diversify the exploration space).
        Defaults to 5.
        :type init_points: int

        :param n_iter: number of steps of bayesian optimization to perform. This starts after the exploration.
        Defaults to 30.
        :type n_iter: int

        :param verbose: whether to print the parameters to explore
        :type verbose: bool

        :param kwargs: the parameter range of the search algorithm
        :return: a dict with the best hyperparameters
        """

        # Default dictionary
        params_to_explore = {
            "batch_size": (100, 500),
            "epochs": (200, 400),
            "coder_dims": (32, 256),
        }

        # If other ranges are provided, they replace the default dictionary values.
        # Only the default keys are authorized - otherwise an error is raised.
        if kwargs:
            for key, value in kwargs.items():
                if key not in params_to_explore.keys():
                    raise ValueError(f"'{key}' is not an authorized parameter")
                params_to_explore[key] = value

        if verbose:
            print("Parameters to explore: ", params_to_explore)

        # The Bayesian Optimization object is created, and the optimization performed.
        tvae_bo = BayesianOptimization(
            self.dist_function, params_to_explore, random_state=9
        )
        tvae_bo.maximize(init_points=init_points, n_iter=n_iter)

        # The best parameters are converted to the actual values (instead of the floats).

        optim_params = {
            "epochs": round(tvae_bo.max["params"]["epochs"]),
            "batch_size": round(tvae_bo.max["params"]["batch_size"]),
            "compress_dims": (
                round(tvae_bo.max["params"]["coder_dims"]),
                round(tvae_bo.max["params"]["coder_dims"]),
            ),
            "decompress_dims": (
                round(tvae_bo.max["params"]["coder_dims"]),
                round(tvae_bo.max["params"]["coder_dims"]),
            ),
        }
        if optim_params["batch_size"] % 2 != 0:
            optim_params["batch_size"] = optim_params["batch_size"] + 1

        return optim_params

    def dist_function(self, batch_size, epochs, coder_dims):
        """
        The metric optimized here is the distinguishability.
        TO DO: allow the loss rather than a metric to be optimized. For that, the fit function
        of the CTGAN would have to be changed (I think) to keep a copy of the history / loss.

        :param batch_size: the size of each training batch
        :type batch_size: float

        :param epochs: the number of training epochs
        :type epochs: float

        :param coder_dims: the size of each hidden layer in the encoder and decoder
        :type coder_dims: float
        """
        # Note that the batch size must be divisible by 2
        params_to_explore = {
            "epochs": round(epochs),
            "batch_size": round(batch_size),
            "compress_dims": (round(coder_dims), round(coder_dims)),
            "decompress_dims": (round(coder_dims), round(coder_dims)),
        }
        if params_to_explore["batch_size"] % 2 != 0:
            params_to_explore["batch_size"] = params_to_explore["batch_size"] + 1

        # run synthesizer training again with given params and get synthetic data
        synthesizer = TVAESynthesizer(self._tvae_metadata, **params_to_explore)
        synthesizer.fit(self._df)
        df_synthetic = synthesizer.sample(num_rows=len(self._df))

        # get metric to optimize
        dist = Distinguishability()
        # block real_df and metadata for now
        metric = dist.compute(self._df, df_synthetic, self._metadata)
        return -(metric["average"]["propensity_mse"])
