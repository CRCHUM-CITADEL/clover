# Standard library
from pathlib import Path
import pytest
import tempfile

# 3rd party packages
import pandas as pd

# Local packages
from metrics.utility.report import Report


test_params = [
    {"nb_cont_columns": i, "nb_cat_columns": j} for i in range(3) for j in range(3)
]
test_ids = [("-").join([f"{k}{v}" for k, v in d.items()]) for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def report(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
) -> Report:
    """
    Compute the report in different settings.

    :param request: the number of continuous and categorical columns to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, split into **train** and **test** sets
    :return: an instance of the report
    """

    metadata = {
        "continuous": ["Clump_Thickness", "Bland_Chromatin"][
            : request.param["nb_cont_columns"]
        ],
        "categorical": ["Class", "Normal_Nucleoli"][: request.param["nb_cat_columns"]],
        "variable_to_predict": "Class",
    }
    if request.param["nb_cat_columns"] == 0:
        metadata["variable_to_predict"] = None

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
        num_repeat=2,
        num_folds=10,
        alpha=0.05,
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

    assert df_summary.shape[1] == 6  # name, objective, min, max, submetric, value


def test_detailed_report(report: Report) -> None:
    """
    Test the detailed report.

    :param report: the computed report fixture
    :return: *None*
    """

    with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated figures
        # Save the figures and check their numbers
        report.detailed(show=False, save_folder=temp_dir, figure_format="png")
        num_figures = len(list(Path(temp_dir).glob("*")))

    num_cont_vars = report.get_num_continuous_variables()
    num_cat_vars = report.get_num_categorical_variables()
    thresh = (
        0 if (num_cont_vars == 0 or num_cat_vars in [0, 1]) else 1
    )  # no figure if there is nothing to report

    assert num_figures >= thresh
