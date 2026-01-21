# Standard library
import pytest
from typing import Type, Tuple
from inspect import getfullargspec
from copy import deepcopy

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local packages
from clover.metrics.base import Metric
from clover.metrics.utility import application as app


def test_fscore() -> None:
    """
    Test the F-score function.

    :return: None
    """

    # Simulate data
    size = 100
    group_ab = np.concatenate(
        [np.random.randint(4, size=size), np.random.randint(6, 10, size=size)]
    )
    group_aa = np.concatenate(
        [np.random.randint(4, size=size), np.random.randint(4, size=size)]
    )
    labels = [0] * size + [1] * size
    labels_wrong = [1] * size + [1] * size

    df_ab = pd.DataFrame.from_dict({"feature": group_ab, "class": labels})
    df_aa = pd.DataFrame.from_dict({"feature": group_aa, "class": labels})
    df_aa_wrong = pd.DataFrame.from_dict({"feature": group_aa, "class": labels_wrong})
    df_labels_only = pd.DataFrame.from_dict({"class": labels})

    with pytest.raises(AssertionError):
        app.FScore.fscore(df=df_ab, predicted_var="")
    with pytest.raises(AssertionError):
        app.FScore.fscore(df=df_ab["feature"], predicted_var="class")
    with pytest.raises(AssertionError):
        app.FScore.fscore(df=df_aa_wrong, predicted_var="class")
    with pytest.raises(AssertionError):
        app.FScore.fscore(df=df_labels_only, predicted_var="class")
    assert all(app.FScore.fscore(df=df_ab, predicted_var="class") > 2)
    assert all(app.FScore.fscore(df=df_aa, predicted_var="class") < 0.05)


test_params = [
    {"metric_class": metric, "which_data": data}
    for metric in app.get_metrics()
    for data in ["different_datasets", "identical_datasets"]
]
test_ids = [f"{d['metric_class'].name}-{d['which_data']}" for d in test_params]


@pytest.fixture(scope="module", params=test_params, ids=test_ids)
def application_metrics_results(
    request,
    df_wbcd: dict[str, pd.DataFrame],
    df_mock_wbcd: dict[str, pd.DataFrame],
    metadata_wbcd: dict,
) -> Tuple[Type[Metric], str, dict]:
    """
    Compute the application metrics in different settings.

    :param request: the number of continuous and categorical columns to test
    :param df_wbcd: the real Wisconsin Breast Cancer Dataset fixture, split into **train** and **test** sets
    :param df_mock_wbcd: the mock wbcd dataset fixture, contains **train** and **test** sets
    :param metadata_wbcd: the wbcd metadata fixture
    :return: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores
    """

    metric_class = request.param["metric_class"]
    which_data = request.param["which_data"]

    # Change the dependent var for testing in both cases
    dependent_var = (
        "Uniformity_of_Cell_Size" if "Regression" in metric_class.name else "Class"
    )
    metadata = deepcopy(metadata_wbcd)
    metadata["variable_to_predict"] = dependent_var

    # Instance parameters
    d = {
        "random_state": 0,
        "num_repeat": 1,
        "num_kfolds": 2,
        "num_optuna_trials": 1,
        "use_gpu": False,
    }

    # Select only the expected instance parameters
    args = getfullargspec(metric_class).args[1:]  # remove self
    metric = metric_class(*[d[arg] for arg in args])

    df_to_compare = df_mock_wbcd if which_data == "different_datasets" else df_wbcd
    scores = metric.compute(df_wbcd, df_to_compare, metadata)

    return metric_class, which_data, scores


def test_application_metrics_summary(
    application_metrics_results: Tuple[Type[Metric], str, dict],
) -> None:
    """
    Test the application metrics average scores.

    :param application_metrics_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = application_metrics_results
    scores = scores["average"]

    for submetric in metric.get_average_submetrics():
        # Check the boundaries
        assert scores[submetric["submetric"]] >= submetric["min"]
        assert scores[submetric["submetric"]] <= submetric["max"]

        # Check the target
        diff_to_objective = abs(
            scores[submetric["submetric"]] - submetric[submetric["objective"]]
        )

        if which_data == "different_datasets":
            assert diff_to_objective > 0.01
        else:
            assert diff_to_objective < 0.01


def test_application_metrics_detailed(
    application_metrics_results: Tuple[Type[Metric], str, dict],
) -> None:
    """
    Test the application metrics detailed scores.

    :param application_metrics_results: a tuple containing the metric class, the dataset type and a dictionary containing
      the **average** scores of the metric and the **detailed** scores

    :return: None
    """

    metric, which_data, scores = application_metrics_results
    report = scores["detailed"]

    metric.draw(report=report, figsize=(8, 6))

    plt.close("all")
