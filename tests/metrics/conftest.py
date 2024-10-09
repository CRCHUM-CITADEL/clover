import pytest  # 3rd party packages
import pandas as pd
import numpy as np


@pytest.fixture(scope="package")
def df_mock_wbcd(df_wbcd: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Generate the continuous mock Wisconsin Breast Cancer Dataset wbcd without ids.

    :param df_wbcd: the wbcd dataset fixture, split into **train** and **test** sets
    :return: the dataframe containing the mock wbcd dataset, split into **train**, **test** and **2nd_gen** sets
    """
    # Shuffle each column with replacement

    df = {}

    for set in ["train", "test"]:
        df[set] = df_wbcd[set].apply(
            lambda x: np.random.choice(x.unique(), size=len(x), replace=True)
        )

    df["2nd_gen"] = df["train"].apply(
        lambda x: np.random.choice(x.unique(), size=len(x), replace=True)
    )

    for set in ["train", "test", "2nd_gen"]:
        # Ensure the support coverage is different
        df[set] = df[set].replace(
            {
                "Clump_Thickness": 3,
                "Uniformity_of_Cell_Shape": 1,
            },
            8,
        )
        # Ensure the consistency is different
        df[set] = df[set].replace({"Bland_Chromatin": 2}, 11)
        df[set] = df[set].replace({"Normal_Nucleoli": "2"}, "11")
        df[set] = df[set].replace({"Uniformity_of_Cell_Shape": 2}, 11)

    return df
