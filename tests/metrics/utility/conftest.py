import pytest  # 3rd party packages
import pandas as pd
import numpy as np


@pytest.fixture(scope="package")
def df_mock_wbcd(df_wbcd: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the continuous mock Wisconsin Breast Cancer Dataset wbcd without ids.

    :param df_wbcd: the wbcd dataset fixture
    :return: the dataframe containing the mock wbcd dataset
    """
    # Shuffle each column with replacement
    df = df_wbcd.apply(
        lambda x: np.random.choice(x.unique(), size=len(x), replace=True)
    )

    # Ensure the support coverage is different
    df = df.replace(
        {
            "Clump_Thickness": 3,
            "Uniformity_of_Cell_Shape": 1,
        },
        8,
    )
    # Ensure the consistency is different
    df = df.replace({"Bland_Chromatin": 2}, 11)
    df = df.replace({"Normal_Nucleoli": "2"}, "11")
    df = df.replace({"Uniformity_of_Cell_Shape": 2}, 11)

    return df
