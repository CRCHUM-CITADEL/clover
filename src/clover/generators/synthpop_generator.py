# Standard library
from typing import Dict, Union, List
from pathlib import Path
import warnings

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from diffprivlib.utils import PrivacyLeakWarning

# Local
from .external.synthpop.synthpop import Synthpop
from .external.dpart.dpart.engines import DPSynthpop
from .base import Generator
from clover.utils.postprocessing import transform_data
import clover.utils.standard as ustandard


class SynthpopGenerator(Generator):
    """
    Wrapper of the Synthpop Python implementation https://github.com/hazy/synthpop
    and a Differentially private version based on the dpart framework https://github.com/hazy/dpart.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param variables_order: the order of the variable to construct sequentially
    :param epsilon: the privacy budget of the differential privacy. One can specify how the budget is split between
        optimizing the visit order (dependency) and training the generator (methods), if prediction_matrix is set to "infer".
        By default, half the budget is allocated for optimization and half for fitting. The budget for training
        each model (for each column) can also be specified. Otherwise, the training budget will be divided equally
        among the columns. For example, epsilon = {"dependency": 0.1, "methods": {"col1": 0.1, "col2": 0.2}...}.
        If epsilon is set to None, a non-DP model will be trained
    :param min_samples_leaf: the minimum number of samples required in a leaf to expand the tree further
        (applicable to non-DP generator)
    :param max_depth: the maximum depth of the tree. If None, the tree expands until all leaves are pure or
        until they contain less than min_samples_leaf samples (applicable to both DP and non-DP generator)
    :param n_bins: number of bins to apply when transforming continuous variables into categorical ones (applicable to DP generator)
    :param methods: defines the specific method each column should be modelled by.
        The default methods to model continuous and discrete columns are DP-Linear Regression
        and DP-Logistic Regression (applicable to DP generator)
    :param prediction_matrix: specify the collection of already visited columns to be used as features
        for each unvisited column. It could be set to "infer" to optimize the variable_order
        by maximizing the information gain. If not None, it will override the variable_order parameter
        (applicable to DP generator)
    :param n_parents: maximum number of columns to be considered as features to predict a target
        (applicable to DP generator)
    :param preprocess_metadata: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening. For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        If not specified, they will be estimated from the real data and a warning will be raised
        (applicable to DP generator)
    """

    name = "Synthpop"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        variables_order: List[str] = None,
        epsilon: Union[float, Dict[str, Union[float, Dict[str, float]]]] = None,
        min_samples_leaf: int = 5,
        max_depth: int = None,
        n_bins: int = 100,
        methods: dict = None,
        prediction_matrix: Union[str, Dict[str, List[str]]] = None,
        n_parents: int = None,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        assert not (
            epsilon is not None and max_depth is None
        ), "max_depth cannot be None when DP is used"

        if preprocess_metadata is None:
            bounds = {}
        else:
            bounds = preprocess_metadata

        self.epsilon = epsilon

        if self.epsilon is None:  # Initiate non-DP generator
            if generator_filepath is None:
                self._gen = Synthpop(
                    visit_sequence=variables_order,
                    seed=random_state,
                    minibucket=min_samples_leaf,
                    max_depth=max_depth,
                )
            self._df = self._df.copy()
            self._dtypes = None
        else:  # Initiate DP generator
            n_col = df.shape[1]
            if n_parents is None:
                self.n_parents = n_col
            else:
                self.n_parents = n_parents

            self.bounds = bounds
            self.bounds_trans = bounds.copy()
            self._df = self._df.copy()
            self.n_bins = n_bins

            # Encoding
            self._scaler = {}
            self._kbins = None

            # New bounds after discretization of continuous variables
            for cont_col in metadata["continuous"]:
                self.bounds_trans[cont_col] = {
                    "categories": list(map(str, list(range(n_bins))))
                }

            # Initialization
            self._df_trans = None

            if generator_filepath is None:
                self._gen = DPSynthpop(
                    methods=methods,
                    epsilon=self.epsilon,
                    bounds=self.bounds_trans,
                    max_depth=max_depth,
                    visit_order=variables_order,
                    prediction_matrix=prediction_matrix,
                    n_parents=self.n_parents,
                )

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        if len(self._metadata["categorical"]) > 0:
            self._df[self._metadata["categorical"]] = self._df[
                self._metadata["categorical"]
            ].astype(
                "category"
            )  # Synthpop requires "category" for categories and not object or str

        if isinstance(self._gen, Synthpop):
            self._dtypes = self._df.dtypes.apply(
                lambda x: x.name.split("64")[0]
            ).to_dict()  # Non-dp generator: only 'int' or 'float' supported without any number after
        else:  # DP mode

            # Initialization
            df_cont = None
            df_cat = None

            if len(self._metadata["continuous"]) > 0:
                # Rescale continuous variables to min and max to prevent privacy leakage
                df_cont_rescaled = pd.DataFrame(columns=self._metadata["continuous"])

                for col, series in self._df[self._metadata["continuous"]].items():
                    if col not in self.bounds:
                        warnings.warn(
                            f"upper and lower bounds not specified for column '{col}'",
                            PrivacyLeakWarning,
                        )
                        self.bounds[col] = {"min": series.min(), "max": series.max()}
                    self._scaler[col] = MinMaxScaler(
                        feature_range=(self.bounds[col]["min"], self.bounds[col]["max"])
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df_cont_rescaled[col] = pd.Series(
                            self._scaler[col].fit_transform(self._df[[col]]).squeeze(),
                            name=col,
                            index=self._df.index,
                        )

                # Convert continuous columns into categorical ones
                self._kbins = KBinsDiscretizer(
                    n_bins=self.n_bins, encode="ordinal", strategy="uniform"
                )
                self._kbins.fit(df_cont_rescaled[self._metadata["continuous"]])
                df_cont = pd.DataFrame(
                    self._kbins.transform(
                        df_cont_rescaled[self._metadata["continuous"]]
                    ),
                    columns=self._metadata["continuous"],
                ).astype("category")

            if len(self._metadata["categorical"]) > 0:
                df_cat = self._df[self._metadata["categorical"]]

            # Merge the preprocessed dataframes
            self._df_trans = pd.concat(
                [df for df in [df_cont, df_cat] if df is not None], axis=1
            )

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Construct the sequential trees (non-DP generator)
        or fit a model for each target column (DP generator).

        :param save_path: the path to save the generator
        :return: *None*
        """

        # Deactivate the package prints while fitting the model
        with ustandard.HiddenPrints():
            if isinstance(self._gen, Synthpop):
                self._gen.fit(self._df, self._dtypes)
            else:
                self._gen.fit(self._df_trans)

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=SynthpopGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the visit order.

        :return: *None*
        """

        if isinstance(self._gen, Synthpop):
            variable_order = list(self._gen.visit_sequence.sort_values().index)

            print("Constructed sequential trees:")
            for i, col in enumerate(variable_order):
                print(f"   {col} has parents {variable_order[:i]}")
        else:
            variable_order = list(self._gen.dep_manager.visit_order)

            print("Variable visit order:")
            for i, col in enumerate(variable_order):
                print(f"   {col} has parents {variable_order[:i]}")

            print("")
            print("Privacy budget spent:")
            print(self._gen.epsilon)

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the models trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        with ustandard.HiddenPrints():  # turn off the prints
            if isinstance(self._gen, Synthpop):
                samples = self._gen.generate(num_samples)
            else:
                samples = self._gen.sample(num_samples)

        # Transform discretized variables to origin continuous ones
        samples = samples[self._df.columns]  # same initial columns order
        if isinstance(self._gen, DPSynthpop):
            if len(self._metadata["continuous"]) > 0:
                samples[self._metadata["continuous"]] = self._kbins.inverse_transform(
                    samples[self._metadata["continuous"]]
                )

                for idx, col in enumerate(self._metadata["continuous"]):
                    half_bin_width = (
                        self._kbins.bin_edges_[idx][1] - self._kbins.bin_edges_[idx][0]
                    ) / 2

                    samples[col] = samples[col] + np.random.uniform(
                        -half_bin_width, half_bin_width, size=len(samples)
                    )

        # Post-processing
        samples = transform_data(
            df_ref=self._df,
            df_to_trans=samples,
            cont_col=self._metadata["continuous"],
        )

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{SynthpopGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
