import pytest  # standard library
from typing import Type, Tuple
import tempfile
from pathlib import Path
from inspect import getfullargspec

# 3rd party packages
import pandas as pd

# Local packages
from generators.base import Generator
from generators.dataSynthesizer import DataSynthesizerGenerator
from generators.synthpop_generator import SynthpopGenerator
from generators.smote import SmoteGenerator
from generators.tvae_generator import TVAEGenerator
from generators.ctgan_generator import CTGANGenerator
from generators.tabddpm_generator import TabDDPMGenerator  # cannot run without GPUs
from generators.mst_generator import MSTGenerator
from generators.ctabgan_generator import CTABGANGenerator


@pytest.mark.parametrize(
    "generator",
    [
        SynthpopGenerator,
        DataSynthesizerGenerator,
        SmoteGenerator,
        TVAEGenerator,
        CTGANGenerator,
        MSTGenerator,
        CTABGANGenerator,
    ],
)
def test_generation(
    generator: Type[Generator], df_wbcd: dict[str, pd.DataFrame], metadata_wbcd: dict
) -> None:
    """
    Check the generation process.

    :param generator: the class of the generator to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param metadata_wbcd: the wbcd metadata fixture
    :return: *None*
    """

    # Instance parameters
    with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated files
        temp_dir = Path(temp_dir)
        datapath = temp_dir / "real_data.csv"
        df_wbcd["train"].to_csv(datapath, index=False)

        d = {
            "df": df_wbcd["train"],
            "metadata": metadata_wbcd,
            "random_state": 0,
            "generator_filepath": None,
            "variables_order": None,  # synthpop
            "min_samples_leaf": 5,  # synthpop
            "max_depth": None,  # synthpop
            "candidate_keys": None,  # datasynthesizer
            "epsilon": 1,  # datasynthesizer / MST / tvae / ctgan
            "max_grad_norm": 1, # tvae / ctgan
            "max_physical_batch_size": 126, # tvae / ctgan
            "degree": 2,  # datasynthesizer
            "k_neighbors": 5,  # smote
            "epochs": 1,  # tvae / ctgan / ctabganplus
            "batch_size": 100,  # tvae / ctgan / tabDDPM / ctabganplus
            "compress_dims": (249, 249),  # tvae
            "decompress_dims": (249, 249),  # tvae
            "discriminator_steps": 2,  # ctgan
            "learning_rate": 1e-5,  # tabDDPM
            "num_timesteps": 2,  # tabDDPM
            "num_iter": 2,  # tabDDPM
            "layers": None,  # tabDDPM
            "delta": 1e-9,  # MST / tvae / ctgan
            "mixed_columns": None,  # ctabganplus
            "log_columns": None,  # ctabganplus
            "integer_columns": None,  # ctabganplus
            "class_dim": (256, 256, 256, 256),  # ctabganplus
            "random_dim": 100,  # ctabganplus
            "num_channels": 64,  # ctabganplus
            "l2scale": 1e-5,  # ctabganplus
        }

        # Select only the expected instance parameters
        args = getfullargspec(generator).args[1:]  # remove self
        gen = generator(*[d[arg] for arg in args])

        # Preprocess and fit the generator
        gen.preprocess()
        gen.fit(save_path=temp_dir)

        # Check that the generator is saved
        num_files = len(list(temp_dir.glob("*")))
        assert (
            num_files >= 2
        ), "The generator should have been saved"  # with the datafile

        # Generate the samples
        df_synth = gen.sample(save_path=temp_dir, num_samples=len(df_wbcd["train"]))

        # Check that the generated samples are consistent
        num_files_plusone = len(list(Path(temp_dir).glob("*")))
        assert num_files_plusone > num_files, "The samples should have been saved"
        assert (
            df_wbcd["train"].shape == df_synth.shape
        ), "Datasets must have the same shape"
        assert set(df_wbcd["train"].columns) == set(
            df_synth.columns
        ), "Datasets must have the same columns"
