# Standard library
from typing import Tuple, Union
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local
from .external.tablediffusion.tablediffusion.models import TableDiffusion_Synthesiser
from .base import Generator
import utils.standard as ustandard


class TableDiffusionGenerator(Generator):
    """
    Wrapper of the differentially private diffusion model for tabular data
    https://github.com/gianlucatruda/TableDiffusion.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param batch_size: the batch size for training
    :param learning_rate: the learning rate for training
    :param dims: related to the output dimension of the residual blocks of the MLP
    :param num_timesteps: the diffusion timesteps for the forward diffusion process
    :param max_grad_norm: the maximum norm of the per-sample gradients.
        Any gradient with norm higher than this will be clipped to this value.
    :param epsilon_target: the privacy budget of the differential privacy
    :param delta_target: target delta to be achieved
    :param epoch_target: number of training epochs you intend to perform
    :param accountant: accounting mechanism, i.e. rdp for RDP accountant and gdp for Gaussian accountant
    :param use_gpu: if to use GPU
    """

    name = "TableDiffusion"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        batch_size: int = None,
        learning_rate: float = None,
        dims: Tuple[int, ...] = None,
        num_timesteps: int = None,
        max_grad_norm: float = None,
        epsilon_target: float = None,
        delta_target: float = None,
        epoch_target: int = None,
        accountant: str = "gdp",
        use_gpu: bool = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        if generator_filepath is None:
            self._gen = TableDiffusion_Synthesiser(
                batch_size=batch_size,
                lr=learning_rate,
                b1=0.5,
                b2=0.999,
                dims=dims,
                diffusion_steps=num_timesteps,
                predict_noise=True,
                max_grad_norm=max_grad_norm,
                epsilon_target=epsilon_target,
                epoch_target=epoch_target,
                delta=delta_target,
                accountant=accountant,
                sample_img_interval=None,
                mlflow_logging=False,
                cuda=use_gpu,
            )

        self._df = self._df.copy()
        self.num_timesteps = num_timesteps
        self.epoch_target = epoch_target
        self.epsilon_target = epsilon_target
        self.delta_target = delta_target
        self._accountant = accountant

        self._original_dtypes = df.dtypes.to_dict()

    def preprocess(self) -> None:
        """
        Prepare the data to train the generator.

        :return: *None*
        """

        self._df[self._metadata["categorical"]] = self._df[
            self._metadata["categorical"]
        ].astype(
            "category"
        )  # Convert categorical features to non-numerical typs; required by TableDiffusion

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Fit a model to predict the added noise at each timestep.

        :param save_path: the path to save the generator
        :return: *None*
        """

        # Deactivate the package prints while fitting the models
        with ustandard.HiddenPrints():
            self._gen.fit(
                df=self._df,
                n_epochs=self.epoch_target,
                epsilon=self.epsilon_target,
                discrete_columns=self._metadata["categorical"],
                verbose=False,
            )

            self.epsilon_spent = self._gen._eps  # Save the used budget

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=TableDiffusionGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the parameters of TableDiffusion.

        :return: *None*
        """

        print("TableDiffusion parameters:")
        print(f"   Number of timesteps: {self.num_timesteps}")
        print(f"   Accounting mechanism: {self._accountant}")
        print(f"   Target epsilon: {self.epsilon_target}")
        print(f"   Target delta: {self.delta_target}")
        print(f"   Spent epsilon: {self.epsilon_spent}")

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the models trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        with ustandard.HiddenPrints():  # turn off the prints
            samples = self._gen.sample(n=num_samples, post_process=True).astype(
                self._original_dtypes
            )

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{TableDiffusionGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
