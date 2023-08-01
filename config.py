from pathlib import Path  # 3rd party packages

# Path configurations
#    Project settings
PROJECT_PATH = Path(".")
DATA_PATH = PROJECT_PATH / "data"
OUTPUT_PATH = PROJECT_PATH / "output"

#   WBCD dataset path from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
WBCD_DATASET_FILEPATH = DATA_PATH / "breast_cancer_wisconsin.csv"
WBCD_DATASET_TRAIN_FILEPATH = DATA_PATH / "breast_cancer_wisconsin_train.csv"
WBCD_DATASET_TEST_FILEPATH = DATA_PATH / "breast_cancer_wisconsin_test.csv"
WBCD_DATASET_FILENAME = "breast_cancer_wisconsin.csv"


# Plot configurations
SEABORN_PALETTE = "YlGnBu_r"
