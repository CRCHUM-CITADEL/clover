# Standard library
import pytest
import tempfile
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local packages
from metrics.metareport import Metareport


@pytest.fixture(scope="module")
def metareport(
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
) -> Metareport:
    """
    Compute the metareport in different settings.

    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, split into **train** and **test** sets
    :return: an instance of the metareport
    """

    metadata = {
        "continuous": ["Clump_Thickness", "Bland_Chromatin"],
        "categorical": ["Class", "Normal_Nucleoli"],
        "variable_to_predict": "Class",
    }

    df_wbcd_mix = {}
    sublist = metadata["continuous"] + metadata["categorical"]
    for set in ["train", "test"]:
        df_wbcd_mix[set] = df_wbcd[set][sublist]

    df_mock = pd.concat(
        [df_mock_wbcd["train"][sublist], df_mock_wbcd["test"][sublist]], axis=0
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        df_mock.to_csv(Path(temp_dir) / "1.csv", index=False)
        df_mock.to_csv(Path(temp_dir) / "2.csv", index=False)

        report = Metareport(
            dataset_name="Wisconsin Breast Cancer Dataset",
            df_real=df_wbcd_mix,
            synthetic_data_path=temp_dir,
            metadata=metadata,
            metrics=["Categorical Consistency"],
        )

        report.compute()

    return report


def test_summary_report(metareport: Metareport) -> None:
    """
    Test the summary metareport.

    :param report: the computed metareport fixture
    :return: *None*
    """
    df_summary = metareport.summary()

    assert (
        df_summary.shape[0] == 2  # the number of datasets to compare
        and df_summary.shape[1] == 1  # the number of metrics computed
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
            metareport_folderpath={"1": temp_dir, "2": temp_dir}
        )  # load

        assert df_summary.equals(
            new_report.summary()
        )  # check the content of the new report
