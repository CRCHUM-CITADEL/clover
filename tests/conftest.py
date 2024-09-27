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

    # Remove the underscores from the column names
    data.rename(columns=lambda x: x.replace("_", ""), inplace=True)

    data["Class"] = (data["Class"] / 2 - 1).astype(
        "int"
    )  # Class 0 or 1 instead of 2 and 4
    data["NormalNucleoli"] = data["NormalNucleoli"].astype(str)  # Categorical variable

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
            "ClumpThickness",
            "UniformityofCellSize",
            "UniformityofCellShape",
            "MarginalAdhesion",
            "SingleEpithelialCellSize",
            "BlandChromatin",
            "Mitoses",
            "BareNuclei",
        ],
        "categorical": ["Class", "NormalNucleoli"],
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
        "ClumpThickness": {"min": 1, "max": 10},
        "UniformityofCellSize": {"min": 1, "max": 10},
        "UniformityofCellShape": {"min": 1, "max": 10},
        "MarginalAdhesion": {"min": 1, "max": 10},
        "SingleEpithelialCellSize": {"min": 1, "max": 10},
        "BlandChromatin": {"min": 1, "max": 10},
        "Mitoses": {"min": 1, "max": 10},
        "BareNuclei": {"min": 1, "max": 10},
        "Class": {"categories": [0, 1]},
        "NormalNucleoli": {
            "categories": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        },
    }

    return preprocess_metadata
