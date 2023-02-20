import pytest  # 3rd party packages
import pandas as pd

import config  # local packages


@pytest.fixture(scope="package")
def df_wbcd() -> pd.DataFrame:
    """
    Load the continuous Wisconsin Breast Cancer Dataset wbcd and delete ids.

    :return: the dataframe containing the wbcd dataset
    """

    df = pd.read_csv(config.WBCD_DATASET_FILEPATH)

    df = df.drop(columns="Sample_code_number")  # identifier not needed

    df["Class"] = (df["Class"] / 2 - 1).astype("int")  # Class 0 or 1 instead of 2 and 4

    df["Normal_Nucleoli"] = df["Normal_Nucleoli"].astype(str)  # Categorical variable

    return df


@pytest.fixture(scope="package")
def metadata_wbcd(df_wbcd: pd.DataFrame) -> dict:
    """
    Return the metadata associating with the Wisconsin Breast Cancer Dataset wbcd dataset.

    :param df_wbcd: the wbcd dataset fixture
    :return: a dict containing the metadata with the following keys:
      **continuous**, **categorical** and **variable_to_predict**
    """

    metadata = {
        "continuous": [
            "Clump_Thickness",
            "Uniformity_of_Cell_Size",
            "Uniformity_of_Cell_Shape",
            "Marginal_Adhesion",
            "Single_Epithelial_Cell_Size",
            "Bland_Chromatin",
            "Mitoses",
            "Bare_Nuclei",
        ],
        "categorical": ["Class", "Normal_Nucleoli"],
        "variable_to_predict": "Class",
    }

    return metadata
