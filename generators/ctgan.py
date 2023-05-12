from typing import Union, List  # standard library

import pandas as pd
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from bayes_opt import BayesianOptimization

from generators.base import Generator  # local
import utils.standard as ustandard
from metrics.utility.population import Distinguishability


class CTGANGenerator(Generator):
    """
    Wrapper of the GAN-based Deep Learning data synthesizer developped by Xu & al (Conditional Tabular GAN).
    https://github.com/sdv-dev

    See article for more information:
    Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
    Modeling tabular data using conditional GAN.
    Advances in Neural Information Processing Systems, 32.
    https://arxiv.org/abs/1907.00503
    """

    name = "CTGAN"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        discriminator_steps=4,
        epochs=300,
        pac=1,
        batch_size=100,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._params = {
            "discriminator_steps": discriminator_steps,
            "epochs": epochs,
            "pac": pac,
            "batch_size": batch_size,
        }
        self._ctgan_metadata = None

        self.preprocess()

        self._gen = (
            CTGANSynthesizer(self._ctgan_metadata, **self._params)
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

        self._ctgan_metadata = SingleTableMetadata.load_from_dict(final_dict)

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._gen.fit(self._df)

        ustandard.save_pickle(
            obj=self._gen, path=save_path, filename=CTGANGenerator.name
        )

    def display(self) -> None:
        """
        Print information about the generator.

        :return: *None*
        """
        print("CTGAN synthesizer parameters: ")
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
            / f"{ustandard.get_date()}_{CTGANGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples

    def search_hyperparameters(self, **kwargs) -> dict:
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

        :param kwargs: a dict containing the parameters of the search algorithm
        :return: a dict with the best hyperparameters
        """

        params_to_explore = {
            "batch_size": (100, 500),
            "discriminator_steps": (1, 8),
            "epochs": (200, 400),
        }

        ctgan_bo = BayesianOptimization(
            self.dist_function, params_to_explore, random_state=9
        )
        ctgan_bo.maximize(**kwargs)

        optim_params = {
            "discriminator_steps": round(ctgan_bo.max["params"]["discriminator_steps"]),
            "epochs": round(ctgan_bo.max["params"]["epochs"]),
            "batch_size": round(
                round(self._params["pac"])
                * (
                    (ctgan_bo.max["params"]["batch_size"] // round(self._params["pac"]))
                    + 1
                )
            ),
        }
        if optim_params["batch_size"] % 2 != 0:
            optim_params["batch_size"] = optim_params["batch_size"] + round(
                self._params["pac"]
            )

        return optim_params

    def dist_function(self, batch_size, epochs, discriminator_steps):
        """
        The metric optimized here is the distinguishability.
        TO DO: allow the loss rather than a metric to be optimized. For that, the fit function
        of the CTGAN would have to be changed (I think) to keep a copy of the history / loss.
        """
        # Note that the batch size must be divisible by 2 and by pac
        params_to_explore = {
            "discriminator_steps": round(discriminator_steps),
            "epochs": round(epochs),
            "batch_size": round(
                round(self._params["pac"])
                * ((batch_size // round(self._params["pac"])) + 1)
            ),
            "pac": self._params["pac"],
        }
        if params_to_explore["batch_size"] % 2 != 0:
            params_to_explore["batch_size"] = params_to_explore["batch_size"] + round(
                self._params["pac"]
            )

        # run synthesizer training again with given params and get synthetic data
        synthesizer = CTGANSynthesizer(self._ctgan_metadata, **params_to_explore)
        synthesizer.fit(self._df)
        df_synthetic = synthesizer.sample(num_rows=len(self._df))

        # get metric to optimize
        dist = Distinguishability()
        # block real_df and metadata for now
        metric = dist.compute(self._df, df_synthetic, self._metadata)
        return -(metric["average"]["propensity_mse"])
