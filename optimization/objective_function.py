# 3rd party packages
import pandas as pd

# Local
import utils.learning as ulearning
from metrics.utility.population import Distinguishability
from metrics.utility.application import Classification, Regression


def distinguishability_hinge_loss(
    df: dict[str, pd.DataFrame],
    df_to_compare: dict[str, pd.DataFrame],
    metadata: dict,
    use_gpu: bool = False,
) -> float:
    """
    The cost or fitness function computed as the Hinge loss applied to the distinguishability metric.

    :param df: the real dataset, split into **train** and **test** sets
    :param df_to_compare: the synthetic dataset, split into **train** and **test** sets
    :param metadata: a dict containing the metadata with the following keys:
      **continuous**, **categorical** and **variable_to_predict**
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    :return: the cost
    """

    # Compute the distinguishability metric
    dist = Distinguishability(num_repeat=20, use_gpu=use_gpu)
    propensity_score = dist.compute(
        df_real=df, df_synthetic=df_to_compare, metadata=metadata
    )["average"]["prediction_mse"]

    # Compute the hinge loss based on the distinguishability score
    loss = ulearning.hinge_loss(propensity_score, threshold=0.05)

    return loss


def absolute_difference_hinge_loss(
    df: dict[str, pd.DataFrame],
    df_to_compare: dict[str, pd.DataFrame],
    metadata: dict,
    use_gpu: bool = False,
) -> float:
    """
    The cost or fitness function computed as the absolute difference
    between real and synthetic scores on the validation set for Classification metric.

    :param df: the real dataset, split into **train** and **test** sets
    :param df_to_compare: the synthetic dataset, split into **train** and **test** sets
    :param metadata: a dict containing the metadata with the following keys:
      **continuous**, **categorical** and **variable_to_predict**
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    :return: the cost
    """

    if metadata["variable_to_predict"] in metadata["categorical"]:
        pred = Classification(num_repeat=20, use_gpu=use_gpu)
    else:
        pred = Regression(num_repeat=20, use_gpu=use_gpu)

    absolute_diff_score = pred.compute(
        df_real=df, df_synthetic=df_to_compare, metadata=metadata
    )["average"]["diff_real_synth"]

    # Compute the hinge loss based on the distinguishability score
    loss = ulearning.hinge_loss(absolute_diff_score, threshold=0.05)

    return loss
