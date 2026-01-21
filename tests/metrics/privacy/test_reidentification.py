# Standard library
import pytest
from typing import Type, Tuple
from inspect import getfullargspec
import math

# 3rd party packages
import pandas as pd
import matplotlib.pyplot as plt

# Local packages
from clover.metrics.base import Metric
from clover.metrics.privacy import reidentification as reid

test_params = [
    {"metric_class": metric, "which_data": data}
    for metric in reid.get_metrics()
    for data in ["different_datasets", "identical_datasets"]
]
test_ids = [f"{d['metric_class'].name}-{d['which_data']}" for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def reidentification_metrics_results(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
) -> Tuple[Type[Metric], str, dict]:
    """
    Compute the reidentification metrics in different settings.

    :param request: the number of continuous and categorical columns to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, contained **train** and **test** sets
    :param metadata_wbcd: the wbcd metadata fixture
    :return: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores
    """

    metric_class = request.param["metric_class"]
    which_data = request.param["which_data"]

    # Instance parameters
    d = {"random_state": 0, "sampling_frac": 1}

    # Select only the expected instance parameters
    args = getfullargspec(metric_class).args[1:]  # remove self
    metric = metric_class(*[d[arg] for arg in args])

    df_to_compare = df_mock_wbcd if which_data == "different_datasets" else df_wbcd
    scores = metric.compute(df_wbcd, df_to_compare, metadata_wbcd)

    return metric_class, which_data, scores


def test_reidentification_metrics_summary(
    reidentification_metrics_results: Tuple[Type[Metric], str, dict],
) -> None:
    """
    Test the reidentification metrics average scores.

    :param reidentification_metrics_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = reidentification_metrics_results
    scores = scores["average"]

    for submetric in metric.get_average_submetrics():
        # Check the boundaries
        assert scores[submetric["submetric"]] >= submetric["min"]
        assert scores[submetric["submetric"]] <= submetric["max"]

        # Filter out the references
        if not submetric["submetric"].endswith("_train_test_ref"):
            # Check the target
            diff_to_objective = abs(
                scores[submetric["submetric"]] - submetric[submetric["objective"]]
            )
            inv_obj = "min" if submetric["objective"] == "max" else "max"
            diff_to_inv_objective = abs(
                scores[submetric["submetric"]] - submetric[inv_obj]
            )

            if which_data == "different_datasets":
                assert (
                    not math.isinf(diff_to_objective) and diff_to_objective < 0.01
                ) or diff_to_inv_objective > 0.01
            else:
                assert (
                    not math.isinf(diff_to_objective) and diff_to_objective > 0.01
                ) or diff_to_inv_objective < 0.01


def test_reidentification_metrics_detailed(
    reidentification_metrics_results: Tuple[Type[Metric], str, dict],
) -> None:
    """
    Test the reidentification metrics detailed scores.

    :param reidentification_metrics_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = reidentification_metrics_results
    report = scores["detailed"]

    metric.draw(report=report, figsize=(8, 6))

    plt.close("all")
