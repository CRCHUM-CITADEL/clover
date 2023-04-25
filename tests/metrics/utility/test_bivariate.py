import pytest  # standard library
from typing import Type, Tuple

import pandas as pd  # 3rd party packages
import numpy as np
import matplotlib.pyplot as plt

from metrics.utility.base import UtilityMetric  # local packages
from metrics.utility import bivariate as biv


test_params = [
    {"metric_class": metric, "which_data": data}
    for metric in biv.get_metrics()
    for data in ["different_datasets", "identical_datasets"]
]
test_ids = [f"{d['metric_class'].name}-{d['which_data']}" for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def bivariate_metric_results(
    request,
    df_wbcd: pd.DataFrame,
    df_mock_wbcd: pd.DataFrame,
    metadata_wbcd: dict,
) -> Tuple[Type[UtilityMetric], str, dict]:
    """
    Compute the bivariate metrics in different settings.

    :param request: the number of continuous and categorical columns to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture
    :param df_mock_wbcd: the mock wbcd dataset fixture
    :param metadata_wbcd: the wbcd metadata fixture
    :return: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores
    """

    metric_class = request.param["metric_class"]
    which_data = request.param["which_data"]

    metric = metric_class()

    df_to_compare = df_mock_wbcd if which_data == "different_datasets" else df_wbcd
    scores = metric.compute(df_wbcd, df_to_compare, metadata_wbcd)

    return metric_class, which_data, scores


def test_bivariate_metrics_summary(
    bivariate_metric_results: Tuple[Type[UtilityMetric], str, dict]
) -> None:
    """
    Test the bivariate metrics average scores.

    :param bivariate_metric_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = bivariate_metric_results
    scores = scores["average"]

    for submetric in metric.get_average_submetrics():
        # Check the boundaries
        assert np.isnan(scores[submetric]) or scores[submetric] >= metric.min
        assert np.isnan(scores[submetric]) or scores[submetric] <= metric.max

        # Check the target
        diff_to_objective = abs(scores[submetric] - getattr(metric, metric.objective))
        if which_data == "different_datasets":
            assert np.isnan(diff_to_objective) or diff_to_objective > 0.01
        else:
            assert np.isnan(diff_to_objective) or diff_to_objective < 0.01


def test_bivariate_metrics_detailed(
    bivariate_metric_results: Tuple[Type[UtilityMetric], str, dict]
) -> None:
    """
    Test the bivariate metrics detailed scores.

    :param bivariate_metric_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = bivariate_metric_results
    report = scores["detailed"]

    metric.draw(report=report, figsize=(8, 6))

    plt.close("all")
