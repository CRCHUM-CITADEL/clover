# Local
import utils.learning as ulearning
from metrics.utility.population import Distinguishability


def distinguishability_hinge_loss(
    df, df_to_compare, metadata, minimize=True, use_gpu=False
) -> float:
    """
    The cost or fitness function computed as the Hinge loss applied to the distinguishability metric.

    :param solution: the solution to evaluate as a numpy array of indices
    :return: the cost
    """

    # Compute the distinguishability metric
    dist = Distinguishability(num_repeat=20, num_folds=0, use_gpu=use_gpu)
    propensity_score = dist.compute(
        df_real=df, df_synthetic=df_to_compare, metadata=metadata
    )["average"]["propensity_mse"]

    # Compute the hinge loss based on the distinguishability score
    loss = ulearning.hinge_loss(propensity_score, threshold=0.05)

    if not minimize:
        loss *= -1

    return loss
