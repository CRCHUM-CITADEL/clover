from typing import Union  # standard library

import pandas as pd  # 3rd party packages
from pathlib import Path

# from sdv.single_table import CTGANSynthesizer
from generators.external.ctgan.single_table.dp_ctgan import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from generators.base import Generator  # local
import utils.standard as ustandard


class CTGANGenerator(Generator):
    """
    Wrapper of the GAN-based Deep Learning data synthesizer developed by Xu & al (Conditional Tabular GAN).
    https://github.com/sdv-dev

    See article for more information:
    Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
    Modeling tabular data using conditional GAN.
    Advances in Neural Information Processing Systems, 32.
    https://arxiv.org/abs/1907.00503

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param discriminator_steps: the number of discriminator updates to do for each generator update.
    :param epochs: the number of training epochs.
    :param batch_size: the batch size for training.
    """

    name = "CTGAN"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        preprocess_metadata: dict = None,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        discriminator_steps: int = 4,
        epochs: int = 300,
        batch_size: int = 100,
        epsilon: float = None,
        preprocess_epsilon_pp: float = None,
        delta: float = None,
        max_grad_norm: float = 1,
        verbose: int = 0,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._params = {
            "discriminator_steps": discriminator_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "delta": delta,
            "epsilon": epsilon,
            "preprocess_epsilon_pp": preprocess_epsilon_pp,
            "max_grad_norm": max_grad_norm,
            "verbose": verbose,
        }
        self._ctgan_metadata = None
        self._preprocess_metadata = preprocess_metadata

        self._gen = (
            None
            if generator_filepath is None
            else ustandard.load_pickle(filepath=generator_filepath)
        )

        if not (
            (epsilon is None and delta is None)
            or (epsilon is not None and delta is not None)
        ):
            raise ValueError(
                "epsilon and delta should either both be specified for differentially private training, "
                "or none should be for non-DP training"
            )

        assert (
            0 <= preprocess_epsilon_pp <= 1
        ), "preprocess_epsilon must be in the interval [0, 1]"

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # The CTGAN library needs a metadata dict
        temp_dict = {}
        final_dict = {}

        for col in self._metadata["continuous"]:
            temp_dict[col] = {"sdtype": "numerical"}
        for col in self._metadata["categorical"]:
            temp_dict[col] = {"sdtype": "categorical"}
        final_dict["columns"] = temp_dict

        self._ctgan_metadata = SingleTableMetadata.load_from_dict(final_dict)

        # The batch size must be divisible by 2 and by pac (=1)
        if self._params["batch_size"] % 2 != 0:
            self._params["batch_size"] = self._params["batch_size"] + 1

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        :param save_path: the path to save the generator
        :return: *None*
        """

        # pac: the number of concatenated samples from one class (real or generated) fed to
        # the discriminator and receiving one label (goal is to mitigate mode collapse) is set to 1.
        pac = 1

        self._gen = CTGANSynthesizer(
            self._ctgan_metadata, self._preprocess_metadata, pac=pac, **self._params
        )
        self._gen.fit(self._df)

        ustandard.save_pickle(
            obj=self._gen, folderpath=save_path, filename=CTGANGenerator.name, date=True
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
