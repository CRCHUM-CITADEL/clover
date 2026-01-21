# Standard library
from pathlib import Path
import pytest
import tempfile

# 3rd party packages
import pandas as pd

# Local packages
from clover.metrics.utility.report import UtilityReport


test_params = [
    {"nb_cont_columns": i, "nb_cat_columns": j} for i in range(3) for j in range(3)
]
test_ids = [("-").join([f"{k}{v}" for k, v in d.items()]) for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def utility_report(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
) -> UtilityReport:
    """
    Compute the utility report in different settings.

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

    report = UtilityReport(
        dataset_name="Wisconsin Breast Cancer Dataset",
        df_real=df_wbcd_mix,
        df_synthetic=df_mock_wbcd_mix,
        metadata=metadata,
        num_repeat=1,
        num_kfolds=3,
        num_optuna_trials=1,
        use_gpu=False,
        alpha=0.05,
    )

    report.compute()

    return report


def test_summary_report(utility_report: UtilityReport) -> None:
    """
    Test the summary report.

    :param utility_report: the computed report fixture
    :return: *None*
    """
    df_summary = utility_report.summary()

    assert (
        df_summary.shape[1] == 7
    )  # name, alias, objective, min, max, submetric, value


def test_detailed_report(utility_report: UtilityReport) -> None:
    """
    Test the detailed report.

    :param utility_report: the computed report fixture
    :return: *None*
    """

    with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated figures
        # Save the figures and check their numbers
        utility_report.detailed(show=False, save_folder=temp_dir, figure_format="png")
        num_figures = len(list(Path(temp_dir).glob("*")))

    num_cont_vars = utility_report.get_num_continuous_variables()
    num_cat_vars = utility_report.get_num_categorical_variables()
    thresh = (
        0 if (num_cont_vars == 0 or num_cat_vars in [0, 1]) else 1
    )  # no figure if there is nothing to report

    assert num_figures >= thresh


def test_save_load_report(utility_report: UtilityReport) -> None:
    """
    Test the save/load operations for the utility report.

    :param utility_report: the computed report fixture
    :return: *None*
    """
    df_summary = utility_report.summary()

    with tempfile.TemporaryDirectory() as temp_dir:
        dir = Path(temp_dir)
        utility_report.save(savepath=temp_dir, filename="report")  # save
        new_report = UtilityReport(report_filepath=dir / "report.pkl")  # load

        assert df_summary.equals(
            new_report.summary()
        )  # check the content of the new report
