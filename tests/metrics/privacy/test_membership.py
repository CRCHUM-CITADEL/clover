# Standard library
import pytest
from typing import Type, Tuple
from inspect import getfullargspec

# 3rd party packages
import pandas as pd
import matplotlib.pyplot as plt

# Local packages
from metrics.base import Metric
from metrics.privacy import membership as mem


test_params = [{"metric_class": metric} for metric in mem.get_metrics()]
test_ids = [d["metric_class"].name for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def membership_metrics_results(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
) -> Tuple[Type[Metric], dict]:
    """
    Compute different membership metrics.

    :param request: requesting test settings
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param metadata_wbcd: the wbcd metadata fixture
    :return: a tuple containing the metric class and a dictionary containing
      the **average** scores of the metric and the **detailed** scores
    """

    metric_class = request.param["metric_class"]

    # Instance parameters
    d = {
        "random_state": 0,
        "sampling_frac": 1,
        "num_repeat": 1,
        "num_kfolds": 2,
        "num_optuna_trials": 1,
        "use_gpu": True,
    }

    # Select only the expected instance parameters
    args = getfullargspec(metric_class).args[1:]  # remove self
    metric = metric_class(*[d[arg] for arg in args])

    # Mock 1st and 2nd generation synthetic data
    df_to_compare = {
        "train": df_wbcd["train"].copy(),
        "test": df_wbcd["test"].copy(),
        "2nd_gen": df_wbcd["train"].copy(),
    }

    # Make half of the synthetic data slightly different, so that around 50% of the data is not just copies
    len_ = len(df_to_compare["train"])
    df_to_compare["train"].loc[: len_ // 2, "Clump_Thickness"] = (
        df_to_compare["train"].loc[: len_ // 2, "Clump_Thickness"] + 1
    )

    scores = metric.compute(df_wbcd, df_to_compare, metadata_wbcd)

    return metric_class, scores


def test_membership_metrics_summary(
    membership_metrics_results: Tuple[Type[Metric], dict]
) -> None:
    """
    Test the membership metrics average scores.

    :param membership_metrics_results: a tuple containing the metric class and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, scores = membership_metrics_results
    scores = scores["average"]

    for submetric in metric.get_average_submetrics():
        # Check the boundaries
        assert scores[submetric["submetric"]] >= submetric["min"]
        assert scores[submetric["submetric"]] <= submetric["max"]


def test_membership_metrics_detailed(
    membership_metrics_results: Tuple[Type[Metric], dict]
) -> None:
    """
    Test the membership metrics detailed scores.

    :param membership_metrics_results: a tuple containing the metric class and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, scores = membership_metrics_results
    report = scores["detailed"]

    metric.draw(report=report, figsize=(8, 6))

    plt.close("all")
