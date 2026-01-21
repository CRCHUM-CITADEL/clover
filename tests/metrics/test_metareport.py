# Standard library
import pytest
import tempfile

# 3rd party packages
import pandas as pd
import numpy as np

# Local packages
from clover.metrics.metareport import Metareport


@pytest.fixture(scope="module")
def metareport(
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
) -> Metareport:
    """
    Compute the metareport in different settings.

    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, split into **train**, **test** and **2nd_gen** sets
    :return: an instance of the metareport
    """

    metadata = {
        "continuous": ["Clump_Thickness", "Bland_Chromatin"],
        "categorical": ["Class", "Normal_Nucleoli"],
        "variable_to_predict": "Class",
    }

    df_wbcd_mix = {}
    df_mock_1 = {}
    df_mock_2 = {}

    sublist = metadata["continuous"] + metadata["categorical"]
    for set_ in ["train", "test"]:
        df_wbcd_mix[set_] = df_wbcd[set_].copy()[sublist]

    for set_ in ["train", "test", "2nd_gen"]:
        df_mock_1[set_] = df_mock_wbcd[set_].copy()[sublist]
        df_mock_2[set_] = (
            df_mock_1[set_]
            .copy()
            .apply(lambda x: np.random.choice(x.unique(), size=len(x), replace=True))
        )

    parameters = {
        "num_repeat": 1,
        "num_kfolds": 2,
        "num_optuna_trials": 1,
        "sampling_frac": 1.0,
        "use_gpu": False,
    }

    report = Metareport(
        dataset_name="Wisconsin Breast Cancer Dataset",
        df_real=df_wbcd_mix,
        synthetic_datasets={"df_mock_1": df_mock_1, "df_mock_2": df_mock_2},
        metadata=metadata,
        random_state=0,
        metrics=["Categorical Consistency", "DCR", "LOGAN"],
        params=parameters,
    )

    report.compute()

    return report


def test_summary_report(metareport: Metareport) -> None:
    """
    Test the summary metareport.

    :param metareport: the computed metareport fixture
    :return: *None*
    """
    df_summary = metareport.summary()

    assert (
        df_summary.shape[0] == 15  # the number of metrics computed
        and df_summary.shape[1] == 2  # the number of datasets to compare
    )


def test_save_load_report(metareport: Metareport) -> None:
    """
    Test the save/load operations for the metareport.

    :param metareport: the computed metareport fixture
    :return: *None*
    """
    df_summary = metareport.summary()

    with tempfile.TemporaryDirectory() as temp_dir:
        metareport.save(savepath=temp_dir)  # save
        new_report = Metareport(
            metareport_folderpath={"df_mock_1": temp_dir, "df_mock_2": temp_dir}
        )  # load

        assert df_summary.equals(
            new_report.summary()
        )  # check the content of the new report
