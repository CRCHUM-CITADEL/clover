import pytest  # standard library
from typing import Type, Tuple, Callable

import pandas as pd  # 3rd party packages
import numpy as np
import matplotlib.pyplot as plt

from metrics.utility.base import UtilityMetric  # local packages
from metrics.utility import univariate as uni
import utils.stats as ustats


@pytest.mark.parametrize(
    "distance_function",
    [
        uni.UnivariateDiscreteDistance.hellinger_distance,
        uni.UnivariateDiscreteDistance.kullback_leibler_divergence,
    ],
)
def test_distance_divergence(distance_function: Callable) -> None:
    """
    Test the distance and divergence functions.

    :param distance_function: the function to test
    :return: None
    """

    p = ustats.discrete_probability_distribution(size=20)
    q = ustats.discrete_probability_distribution(size=20)

    with pytest.raises(AssertionError):
        distance_function(p, np.array([]))
    with pytest.raises(AssertionError):
        distance_function(np.array([0.6, 0.8]), np.array([0.3, 0.7]))

    assert distance_function(p, q) > 0.1
    assert distance_function(p, p) == 0


test_params = [
    {"metric_class": metric, "which_data": data}
    for metric in uni.get_metrics()
    for data in ["different_datasets", "identical_datasets"]
]
test_ids = [f"{d['metric_class'].name}-{d['which_data']}" for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def univariate_metric_results(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
) -> Tuple[Type[UtilityMetric], str, dict]:
    """
    Compute the univariate metrics in different settings.

    :param request: the number of continuous and categorical columns to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, split into **train** and **test** sets
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


def test_univariate_metrics_summary(
    univariate_metric_results: Tuple[Type[UtilityMetric], str, dict]
) -> None:
    """
    Test the univariate metrics average scores.

    :param univariate_metric_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = univariate_metric_results
    scores = scores["average"]

    for submetric in metric.get_average_submetrics():
        # Check the boundaries
        assert scores[submetric] >= metric.min
        assert scores[submetric] <= metric.max

        # Check the target
        diff_to_objective = abs(scores[submetric] - getattr(metric, metric.objective))
        if which_data == "different_datasets":
            assert diff_to_objective > 0.01
        else:
            assert diff_to_objective < 0.01


def test_univariate_metrics_detailed(
    univariate_metric_results: Tuple[Type[UtilityMetric], str, dict]
) -> None:
    """
    Test the univariate metrics detailed scores.

    :param univariate_metric_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = univariate_metric_results
    report = scores["detailed"]

    metric.draw(report=report, figsize=(8, 6))

    plt.close("all")
