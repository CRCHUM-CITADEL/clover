# Standard library
import pytest
import tempfile
from typing import Callable

# 3rd party packages
import pandas as pd

# Local packages
from metrics.report import Report

test_params = [["DCR", "Categorical Consistency"], ["Categorical Consistency"]]


@pytest.fixture(
    scope="module", params=test_params, ids=[f"{len(p)}metric(s)" for p in test_params]
)
def report(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
) -> Report:
    """
    Compute the generic report in different settings.

    :param request: the list of metrics to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, split into **train** and **test** sets
    :return: an instance of the report
    """

    metadata = {
        "continuous": ["Clump_Thickness", "Bland_Chromatin"],
        "categorical": ["Class", "Normal_Nucleoli"],
        "variable_to_predict": "Class",
    }

    df_wbcd_mix = {}
    df_mock_wbcd_mix = {}
    for set in ["train", "test"]:
        df_wbcd_mix[set] = df_wbcd[set][
            metadata["continuous"] + metadata["categorical"]
        ]
        df_mock_wbcd_mix[set] = df_mock_wbcd[set][
            metadata["continuous"] + metadata["categorical"]
        ]

    report = Report(
        dataset_name="Wisconsin Breast Cancer Dataset",
        df_real=df_wbcd_mix,
        df_synthetic=df_mock_wbcd_mix,
        metadata=metadata,
        metrics=request.param,
        params={"sampling_frac": 0.5},
    )

    report.compute()

    return report


def test_summary_report(report: Report) -> None:
    """
    Test the summary report.

    :param report: the computed report fixture
    :return: *None*
    """
    df_summary = report.summary()

    assert (
        df_summary.shape[1] == 7
    )  # name, alias, objective, min, max, submetric, value


def test_save_load_report(report: Report) -> None:
    """
    Test the save/load operations for the generic report.

    :param report: the computed report fixture
    :return: *None*
    """
    df_summary = report.summary()

    with tempfile.TemporaryDirectory() as temp_dir:
        report.save(savepath=temp_dir, filename="report")  # save
        new_report = Report(
            report_folderpath=temp_dir, report_filename="report"
        )  # load

        assert df_summary.equals(
            new_report.summary()
        )  # check the content of the new report


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        ["Unknown"],
    ],
    ids=["no-metric", "unknown"],
)
def test_assertion_error_metrics(metrics: list[str]) -> None:
    """
    Test that an exception is raised when no or a wrong metric is provided.

    :param metrics: the metrics to test
    :return: *None*
    """

    with pytest.raises(AssertionError):
        Report(
            dataset_name="Wisconsin Breast Cancer Dataset",
            df_real={"": pd.DataFrame([])},
            df_synthetic={"": pd.DataFrame([])},
            metadata={},
            metrics=metrics,
        )
