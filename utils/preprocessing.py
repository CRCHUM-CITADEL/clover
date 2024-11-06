from typing import Tuple  # standard library

import pandas as pd  # 3rd party packages
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def bin_per_column(
    df_ref: pd.DataFrame, df_tobin: pd.DataFrame, bin_size: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin a continuous dataframe and its reference variable per variable.

    :param df_ref: the reference dataframe to bin
    :param df_tobin: another dataframe to bin
    :param bin_size: the number of bins
    :return: the binned reference and application dataframes
    """
    df_ref_bin = {}
    df_tobin_bin = {}

    # Bin each column independently
    for col in df_ref.columns:
        # Fit the bins on the reference dataframe and apply them
        kbins = KBinsDiscretizer(n_bins=bin_size, encode="ordinal", strategy="uniform")
        df_ref_bin[col] = kbins.fit_transform(df_ref[[col]])[:, 0]
        bin_edges = kbins.bin_edges_[0]  # only 1 column
        # In case the min max of the dataframe to bin are greater
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        # Bin the dataframe with the reference bins
        df_tobin_bin[col] = pd.cut(
            df_tobin[col], bins=bin_edges, labels=np.arange(bin_size)
        ).to_numpy()

    df_ref_bin = pd.DataFrame.from_dict(df_ref_bin).astype(int).astype(str)
    df_tobin_bin = pd.DataFrame.from_dict(df_tobin_bin).astype(int).astype(str)

    return df_ref_bin, df_tobin_bin


def range_query(series: pd.Series, a: float or int, b: float or int) -> int:
    """
    Performs a range query on a DataFrame.

    :param series: The pandas Series to apply the range query to.
    :param a: The lower bound of the range (inclusive).
    :param b: The upper bound of the range (exclusive).

    :return: The number of rows in the DataFrame where the values in the specified column
        fall within the range [a, b).
    """
    return sum((series >= a) & (series < b))


def laplace_mech(
    v: int or float, sensitivity: int or float, epsilon: int or float
) -> float:
    """
    Return a value with noise added using the Laplace mechanism.

    :param v: The input value.
    :param sensitivity: The sensitivity of the function.
    :param epsilon: The privacy parameter.

    :return: The input value(s) with Laplace noise added.
    """
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)


def generate_continuous_dp(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    epsilon: float,
    modes=None,
    sensitivity=1,
    num_bins=None,
    decimals=None,
):
    """
    Generate differentially private continuous samples from a DataFrame.

    :param df: The DataFrame containing the data.
    :param col: The column in the DataFrame to generate samples from.
    :param min_val: The minimum value of the range.
    :param max_val: The maximum value of the range.
    :param epsilon: The privacy parameter for differential privacy.
    :param modes: List of modes that should not be synthesized as they are specified by the user.
    :param sensitivity: float or int
        The sensitivity parameter. Defaults to 1.
    :param num_bins: int, optional
        The number of bins to discretize the range. If not provided, it's calculated based on the range.
    :param decimals: int, optional
        The number of decimal places to round the generated samples.
        If not provided, the values are rounded to the nearest integer.

    :return: array_like
        An array of differentially private continuous samples.
    """
    assert max_val > min_val

    # Calculate num_bins if not provided
    if num_bins is None:
        # num_bins = int(max(round(max_val - min_val, 0), 10))
        num_bins = 100

    # Exclude modes if specified by the user
    if modes is not None:
        cleaned_series = df[~df[col].isin(modes)][col].copy()
        modes_series = df[df[col].isin(modes)][col].copy()
    else:
        cleaned_series = df[col].copy()

    # Calculate bin width and generate bins
    bin_width = (max_val - min_val) / num_bins
    bins = np.linspace(min_val, max_val, num_bins, endpoint=False)

    # Calculate counts for each bin
    counts = [range_query(cleaned_series, b, b + bin_width) for b in bins]

    # Apply Laplace mechanism and normalize counts
    dp_syn_rep = [max(0, laplace_mech(c, sensitivity, epsilon)) for c in counts]
    syn_normalized = dp_syn_rep / np.sum(dp_syn_rep)

    # Sample from bins based on synthetic representation
    samples = np.random.choice(bins, len(cleaned_series), p=syn_normalized)
    samples_uni = [np.random.uniform(value, value + bin_width) for value in samples]

    if modes is not None:
        samples_uni = pd.concat(
            [samples_uni, pd.concat([modes_series], ignore_index=True)]
        )

    # if decimals is not None:
    #    return np.round(samples_uni, decimals)

    else:
        return samples_uni
