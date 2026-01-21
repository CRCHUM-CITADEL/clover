import pytest  # standard library
from typing import Type
import tempfile
from pathlib import Path
from inspect import getfullargspec

# 3rd party packages
import pandas as pd

# Local packages
from clover.generators import (
    Generator,
    DataSynthesizerGenerator,
    SynthpopGenerator,
    SmoteGenerator,
    TVAEGenerator,
    CTGANGenerator,
    FinDiffGenerator,
    MSTGenerator,
    CTABGANGenerator,
)


@pytest.mark.parametrize(
    "generator_non_dp",
    [
        DataSynthesizerGenerator,
        SynthpopGenerator,
        SmoteGenerator,
        CTGANGenerator,
        TVAEGenerator,
        CTABGANGenerator,
        FinDiffGenerator,
    ],
)
def test_generation_non_dp(
    generator_non_dp: Type[Generator],
    df_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
) -> None:
    """
    Check the generation process for non differentially private generators.

    :param generator_non_dp: the class of the generator to test
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
            "epsilon": None,
            "candidate_keys": None,  # datasynthesizer
            "degree": 2,  # datasynthesizer
            "variables_order": None,  # synthpop
            "min_samples_leaf": 5,  # synthpop
            "max_depth": None,  # synthpop
            "k_neighbors": 5,  # smote
            "discriminator_steps": 2,  # ctgan
            "epochs": 2,  # ctgan / tvae / ctabganplus / findiff
            "batch_size": 100,  # ctgan / tvae / ctabganplus / findiff
            "verbose": 0,  # ctgan
            "max_physical_batch_size": 126,  # tvae
            "compress_dims": (249, 249),  # tvae
            "decompress_dims": (249, 249),  # tvae
            "mixed_columns": None,  # ctabganplus
            "log_columns": None,  # ctabganplus
            "integer_columns": None,  # ctabganplus
            "class_dim": (256, 256, 256, 256),  # ctabganplus
            "random_dim": 100,  # ctabganplus
            "num_channels": 64,  # ctabganplus
            "l2scale": 1e-5,  # ctabganplus
            "learning_rate": 1e-4,  # findiff
            "diffusion_steps": 50,  # findiff
            "mpl_layers": [1024, 1024, 1024, 1024],  # findiff
            "activation": "lrelu",  # findiff
            "dim_t": 64,  # findiff
            "cat_emb_dim": 2,  # findiff
            "diff_beta_start_end": [1e-4, 0.02],  # findiff
            "scheduler": "linear",  # findiff
            # Parameters not applicable to non-dp generators
            "preprocess_metadata": None,
            "n_bins": None,  # synthpop
            "methods": None,  # synthpop
            "prediction_matrix": None,  # synthpop
            "n_parents": None,  # synthpop
            "nu": None,  # smote
            "r": None,  # smote
            "delta": None,  # ctgan / tvae / ctabganplus / findiff
            "max_grad_norm": None,  # ctgan / tvae / ctabganplus / findiff
            "nbins": 10,  # mst
        }

        # Select only the expected instance parameters
        args = getfullargspec(generator_non_dp).args[1:]  # remove self
        gen = generator_non_dp(*[d[arg] for arg in args])

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


@pytest.mark.parametrize(
    "generator_dp",
    [
        DataSynthesizerGenerator,
        SynthpopGenerator,
        SmoteGenerator,
        MSTGenerator,
        CTGANGenerator,
        TVAEGenerator,
        CTABGANGenerator,
        FinDiffGenerator,
    ],
)
def test_generation_dp(
    generator_dp: Type[Generator],
    df_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
    preprocess_metadata_wbcd: dict,
) -> None:
    """
    Check the generation process for differentially private generators.

    :param generator_dp: the class of the generator to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param metadata_wbcd: the wbcd metadata fixture
    :param preprocess_metadata_wbcd: the wbcd preprocessing metadata fixture
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
            "epsilon": 1,
            "preprocess_metadata": preprocess_metadata_wbcd,
            "candidate_keys": None,  # datasynthesizer
            "degree": 2,  # datasynthesizer
            "variables_order": None,  # synthpop
            "max_depth": 3,  # synthpop
            "n_bins": 10,  # synthpop
            "methods": None,  # synthpop
            "prediction_matrix": None,  # synthpop
            "n_parents": 2,  # synthpop
            "k_neighbors": 5,  # smote
            "nu": 0.5,  # smote
            "cat_emb_dim": 2,  # smote / findiff
            "r": 1,  # smote
            "delta": 1e-9,  # MST / ctgan / tvae / ctabganplus / findiff
            "discriminator_steps": 2,  # ctgan
            "epochs": 2,  # ctgan / tvae / ctabganplus / findiff
            "batch_size": 100,  # ctgan / tvae / ctabganplus / findiff
            "max_grad_norm": 1,  # ctgan / tvae / ctabganplus / findiff
            "verbose": 0,  # ctgan
            "max_physical_batch_size": 126,  # tvae
            "compress_dims": (249, 249),  # tvae
            "decompress_dims": (249, 249),  # tvae
            "mixed_columns": None,  # ctabganplus
            "log_columns": None,  # ctabganplus
            "integer_columns": None,  # ctabganplus
            "class_dim": (256, 256, 256, 256),  # ctabganplus
            "random_dim": 100,  # ctabganplus
            "num_channels": 64,  # ctabganplus
            "l2scale": 1e-5,  # ctabganplus
            "learning_rate": 1e-4,  # findiff
            "diffusion_steps": 50,  # findiff
            "mpl_layers": [1024, 1024, 1024, 1024],  # findiff
            "activation": "lrelu",  # findiff
            "dim_t": 64,  # findiff
            "diff_beta_start_end": [1e-4, 0.02],  # findiff
            "scheduler": "linear",  # findiff
            # Parameters not applicable to dp generators
            "min_samples_leaf": None,  # synthpop
            "nbins": 10,  # mst
        }

        # Select only the expected instance parameters
        args = getfullargspec(generator_dp).args[1:]  # remove self
        gen = generator_dp(*[d[arg] for arg in args])

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
