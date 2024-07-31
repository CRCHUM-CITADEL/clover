from typing import Union, Tuple  # standard library

import pandas as pd  # 3rd party packages
from pathlib import Path
from generators.external.ctgan.single_table.dp_ctgan import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from generators.base import Generator  # local
import utils.standard as ustandard


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

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param epochs: the number of training epochs.
    :param batch_size: the batch size for training.
    :param compress_dims: the size of the hidden layers in the encoder.
    :param decompress_dims: the size of the hidden layers in the decoder.
    """

    name = "TVAE"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        preprocess_metadata: dict = None,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        epochs: int = 300,
        batch_size: int = 100,
        compress_dims: Tuple[int, int] = (249, 249),
        decompress_dims: Tuple[int, int] = (249, 249),
        epsilon: float = None,
        preprocess_epsilon_pp: float = None,
        delta: float = None,
        max_grad_norm: float = 1,
        max_physical_batch_size: int = 125,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        self._params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "compress_dims": compress_dims,
            "decompress_dims": decompress_dims,
            "delta": delta,
            "epsilon": epsilon,
            "preprocess_epsilon_pp": preprocess_epsilon_pp,
            "max_grad_norm": max_grad_norm,
            "max_physical_batch_size": max_physical_batch_size,
        }
        self._tvae_metadata = None
        self._preprocess_metadata = preprocess_metadata

        if not (
            (epsilon is None and delta is None)
            or (epsilon is not None and delta is not None)
        ):
            raise ValueError(
                "epsilon and delta should either both be specified for differentially private training, "
                "or none should be for non-DP training"
            )

        assert (
            preprocess_epsilon_pp is None or 0 <= preprocess_epsilon_pp <= 1
        ), "preprocess_epsilon must be in the interval [0, 1]"

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # The library needs a metadata dict
        temp_dict = {}
        final_dict = {}

        for col in self._metadata["continuous"]:
            temp_dict[col] = {"sdtype": "numerical"}
        for col in self._metadata["categorical"]:
            temp_dict[col] = {"sdtype": "categorical"}
        final_dict["columns"] = temp_dict

        self._tvae_metadata = SingleTableMetadata.load_from_dict(final_dict)

        # The batch size must be divisible by 2
        if self._params["batch_size"] % 2 != 0:
            self._params["batch_size"] = self._params["batch_size"] + 1

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Train the generator and save it.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._gen = TVAESynthesizer(
            self._tvae_metadata, self._preprocess_metadata, **self._params
        )

        self._gen.fit(self._df)

        # necessary to be able to pickle the model
        if self._params["epsilon"] is not None and self._params["delta"] is not None:
            self._gen._model.vae.remove_hooks()

        ustandard.save_pickle(
            obj=self._gen, folderpath=save_path, filename=TVAEGenerator.name, date=True
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
