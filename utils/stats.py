from typing import Tuple  # standard library

import pandas as pd  # 3rd party packages
import numpy as np
import scipy.stats


def frequency_table(s: pd.Series, name: str) -> pd.DataFrame:
    """
    Create a pandas dataframe frequency table.

    :param s: the pandas series
    :param name: the input name used for naming the columns
    :return: a dataframe containing the value counts and the value frequencies of the input
    """

    df = s.value_counts().to_frame(name=f"{name}_count")
    df[f"{name}_freq"] = df[f"{name}_count"] / len(s)

    return df


def discrete_probability_distribution(size: int) -> np.ndarray:
    """
    Generate a random discrete probability distribution.

    :param size: the size of the vector
    :return: the probability distribution
    """

    p = np.random.random(size=size)
    p /= np.sum(p)

    return p


def scipy_chi2_contingency(
    predictor: np.ndarray, response: np.ndarray
) -> Tuple[float, float, int, np.ndarray]:
    """
    Chi-square test of independence of variables in a contingency table (see https://docs.scipy.org).

    :param predictor: the predictor or independent variable
    :param response: the response or dependent variable
    :return: the test statistic, the p-value, the degrees of freedom and the expected frequencies
    """

    contingency_table = pd.crosstab(predictor, response)

    chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)

    return chi2, p, dof, expected
