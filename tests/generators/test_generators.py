import pytest  # standard library
from typing import Type, Tuple
import tempfile
from pathlib import Path
from inspect import getfullargspec

import pandas as pd  # 3rd party packages

from generators.base import Generator  # local packages
from generators.dataSynthesizer import DataSynthesizerGenerator
from generators.synthpop_generator import SynthpopGenerator


@pytest.mark.parametrize("generator", [SynthpopGenerator, DataSynthesizerGenerator])
def test_generation(
    generator: Type[Generator], df_wbcd: pd.DataFrame, metadata_wbcd: dict
) -> None:
    """
    Check the generation process.

    :param generator: the class of the generator to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture
    :param metadata_wbcd: the wbcd metadata fixture
    :return: *None*
    """

    # Instance parameters
    with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated files
        datapath = Path(temp_dir) / "real_data.csv"
        df_wbcd.to_csv(datapath, index=False)

        d = {
            "df": df_wbcd,
            "metadata": metadata_wbcd,
            "generator_filepath": None,
            "variables_order": None,  # synthpop
            "candidate_keys": None,  # datasynthesizer
            "epsilon": 0,  # datasynthesizer
            "degree": 2,  # datasynthesizer
        }

        # Select only the expected instance parameters
        args = getfullargspec(generator).args[1:]  # remove self
        gen = generator(*[d[arg] for arg in args])

        # Preprocess and fit the generator
        gen.preprocess()
        gen.fit(save_path=temp_dir)

        # Check that the generator is saved
        num_files = len(list(Path(temp_dir).glob("*")))
        assert (
            num_files == 2
        ), "The generator should have been saved"  # with the datafile

        # Generate the samples
        df_synth = gen.sample(save_path=temp_dir, num_samples=len(df_wbcd))

        # Check that the generated samples are consistent
        num_files = len(list(Path(temp_dir).glob("*")))
        assert num_files == 3, "The samples should have been saved"
        assert df_wbcd.shape == df_synth.shape, "Datasets must have the same shape"
        assert set(df_wbcd.columns) == set(
            df_synth.columns
        ), "Datasets must have the same columns"

        # Check the hyperparameters search method
        hyper = gen.search_hyperparameters(num_iter=1)
        assert isinstance(hyper, dict)
