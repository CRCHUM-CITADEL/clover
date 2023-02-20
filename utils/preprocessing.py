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
