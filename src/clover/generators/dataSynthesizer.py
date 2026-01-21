from typing import Union, List, Dict  # standard library
from pathlib import Path
import tempfile
import warnings
import json

import pandas as pd  # 3rd party packages
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from diffprivlib.utils import PrivacyLeakWarning

from clover.generators.base import Generator  # local
from clover.utils.postprocessing import transform_data
import clover.utils.standard as ustandard


class DataSynthesizerGenerator(Generator):
    """
    Wrapper of the DataSynthesizer implementation https://github.com/DataResponsibly/DataSynthesizer.

    See the article `Ping, Haoyue, Julia Stoyanovich, and Bill Howe.
    "Datasynthesizer: Privacy-preserving synthetic datasets." Proceedings of the 29th International Conference on
    Scientific and Statistical Database Management. 2017.
    <https://dl.acm.org/doi/abs/10.1145/3085504.3091117>`_ for more details.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param candidate_keys: the candidate keys of the original database
    :param epsilon: the epsilon-DP for the Differential Privacy (0 for no added noise)
    :param degree: the maximum numbers of parents in the Bayesian Network
    :param preprocess_metadata: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening. For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        If not specified, they will be estimated from the real data and a warning will be raised
        (for differentially private model)
    """

    name = "DataSynthesizer"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        candidate_keys: List[str] = None,
        epsilon: float = None,
        degree: int = 5,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, None)

        bounds = preprocess_metadata

        self._attribute_to_is_categorical = {}
        self._candidate_keys = candidate_keys if candidate_keys is not None else []
        self._attribute_to_is_candidate_key = {}
        self._threshold = None
        self._degree = degree
        self._generator_filepath = generator_filepath

        if epsilon is None:
            self._epsilon = 0  # No differential privacy
        else:
            self._epsilon = epsilon

        if bounds is None:
            bounds = {}

        self.categorical_attribute_domain = {}
        self.numerical_attribute_ranges = {}

        for col in self._metadata["categorical"]:
            if col in bounds:
                self.categorical_attribute_domain["col"] = bounds[col]["categories"]
            else:
                if self._epsilon != 0:
                    warnings.warn(
                        f"List of categories not specified for column '{col}', categories will be extracted from real data for this variable",
                        PrivacyLeakWarning,
                    )

        for col in self._metadata["continuous"]:
            if col in bounds:
                self.numerical_attribute_ranges["col"] = [
                    bounds[col]["min"],
                    bounds[col]["max"],
                ]
            else:
                if self._epsilon != 0:
                    warnings.warn(
                        f"upper and lower bounds not specified for column '{col}'",
                        PrivacyLeakWarning,
                    )

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # Max numbers of categories; this decides if a variable not specified as categorical can be treated as such
        self._threshold = self._df.value_counts().max()

        # Fill categorical vars and candidate keys dictionaries
        for col in self._df.columns:
            self._attribute_to_is_categorical[col] = (
                col in self._metadata["categorical"]
            )
            self._attribute_to_is_candidate_key[col] = col in self._candidate_keys

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Construct the Bayesian network.

        :param save_path: the path to save the generator
        :return: *None*
        """

        with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the data
            datapath = Path(temp_dir) / "real_data.csv"
            self._df.to_csv(datapath, index=False)

            categorical_attribute_domain_file = (
                f"{temp_dir}/categorical_attribute_domain.json"
            )

            with open(categorical_attribute_domain_file, "w") as json_file:
                json.dump(self.categorical_attribute_domain, json_file)

            with ustandard.HiddenPrints():  # turn off the prints
                describer = DataDescriber(category_threshold=self._threshold)
                describer.describe_dataset_in_correlated_attribute_mode(
                    dataset_file=str(datapath),
                    epsilon=self._epsilon,
                    k=self._degree,
                    attribute_to_is_categorical=self._attribute_to_is_categorical,
                    attribute_to_is_candidate_key=self._attribute_to_is_candidate_key,
                    categorical_attribute_domain_file=categorical_attribute_domain_file,
                    numerical_attribute_ranges=self.numerical_attribute_ranges,
                    seed=self._random_state,
                )

            self._generator_filepath = (
                Path(save_path)
                / f"{ustandard.get_date()}_{DataSynthesizerGenerator.name}_epsilon{self._epsilon}.json"
            )
            describer.save_dataset_description_to_file(self._generator_filepath)

    def display(self) -> None:
        """
        Print the constructed Bayesian network.

        :return: *None*
        """

        describer = read_json_file(self._generator_filepath)
        display_bayesian_network(describer["bayesian_network"])

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the Bayesian network trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        assert self._generator_filepath is not None, "No generator to sample from"

        generator = DataGenerator()
        # Correlated mode: more computationally expensive but capture relations between variables
        generator.generate_dataset_in_correlated_attribute_mode(
            n=num_samples,
            description_file=self._generator_filepath,
            seed=self._random_state,
        )

        # Read the data after saving - not returned by the package
        filepath = (
            Path(save_path)
            / f"{ustandard.get_date()}_{DataSynthesizerGenerator.name}_{num_samples}samples.csv"
        )
        generator.save_synthetic_data(filepath)
        samples = pd.read_csv(filepath)

        # Post-processing
        samples = transform_data(
            df_ref=self._df,
            df_to_trans=samples,
            cont_col=self._metadata["continuous"],
        )

        # Re-save the processed data
        samples.to_csv(filepath, index=False)

        return samples
