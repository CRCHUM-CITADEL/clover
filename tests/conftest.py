import pytest  # 3rd party packages
import pandas as pd

import config  # local packages


@pytest.fixture(scope="package")
def df_wbcd() -> dict[str, pd.DataFrame]:
    """
    Load the continuous Wisconsin Breast Cancer Dataset wbcd and delete ids.

    :return: the dataframe containing the wbcd dataset, split into **train** and **test** sets
    """

    data = pd.read_csv(config.WBCD_DATASET_FILEPATH)
    data = data.drop(columns="Sample_code_number")  # identifier not needed

    data["Class"] = (data["Class"] / 2 - 1).astype(
        "int"
    )  # Class 0 or 1 instead of 2 and 4
    data["Normal_Nucleoli"] = data["Normal_Nucleoli"].astype(
        str
    )  # Categorical variable

    # Split train / test
    df = {}
    df["train"] = data.sample(frac=0.8, replace=False, random_state=66)
    df["test"] = data.drop(index=df["train"].index).reset_index(drop=True)
    df["train"] = df["train"].reset_index(drop=True)

    return df


@pytest.fixture(scope="package")
def metadata_wbcd() -> dict:
    """
    Return the metadata associating with the Wisconsin Breast Cancer Dataset wbcd dataset.

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


@pytest.fixture(scope="package")
def preprocess_metadata_wbcd() -> dict:
    """
    Return the preprocessing metadata associating with the Wisconsin Breast Cancer Dataset wbcd dataset.

    :return: a dict containing the preprocessing metadata for training differentially private generator
    """

    preprocess_metadata = {
        "Clump_Thickness": {"min": 1, "max": 10},
        "Uniformity_of_Cell_Size": {"min": 1, "max": 10},
        "Uniformity_of_Cell_Shape": {"min": 1, "max": 10},
        "Marginal_Adhesion": {"min": 1, "max": 10},
        "Single_Epithelial_Cell_Size": {"min": 1, "max": 10},
        "Bland_Chromatin": {"min": 1, "max": 10},
        "Mitoses": {"min": 1, "max": 10},
        "Bare_Nuclei": {"min": 1, "max": 10},
        "Class": {"categories": [0, 1]},
        "Normal_Nucleoli": {
            "categories": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        },
    }

    return preprocess_metadata
