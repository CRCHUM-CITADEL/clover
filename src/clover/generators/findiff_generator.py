# Standard library
from typing import Dict, Union
from pathlib import Path
from functools import partialmethod
import warnings

# 3rd party packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from diffprivlib.utils import PrivacyLeakWarning

# Local
from .base import Generator
from .external.findiff.findiff import FinDiff
from clover.utils.postprocessing import transform_data
import clover.utils.standard as ustandard

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def generate_synth_var(
    var: pd.Series, min: float, max: float, epsilon: float, n: int, n_bins: int = 100
) -> np.ndarray:
    """Generate synthetic data for a single variable with differential privacy

    :param var: the real data
    :param min: minimum value of the real data (must not be provided by looking at the real data to prevent privacy leakage)
    :param max: maximum value of the real data (must not be provided by looking at the real data)
    :param epsilon: privacy budget
    :param n: length of the synthetic data
    :param n_bins: number of bins for the histogram
    :return: synthetic data
    """

    bins = np.linspace(min, max, n_bins)
    bin_size = (max - min) / (n_bins - 1)

    counts = [len(var[(var >= b) & (var < b + bin_size)]) for b in bins]

    dp_syn_rep = [c + np.random.laplace(loc=0, scale=1 / epsilon) for c in counts]

    # Remove negative count
    dp_syn_rep_nn = np.clip(dp_syn_rep, 0, None)

    syn_normalized = dp_syn_rep_nn / np.sum(dp_syn_rep_nn)

    return np.random.choice(bins, n, p=syn_normalized)


class FinDiffGenerator(Generator):
    """
    Wrapper of the tabular diffusion models FinDiff https://github.com/sattarov/FinDiff.
    The original model was modified to implement differential privacy.

    Note: The authors of the original paper tested the model on three datasets, the smallest of which, Credit Default,
    contains 30,000 observations. If the model does not function optimally with your available observations, consider
    augmenting your data.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
        (make sure the column name does not contain underscore)
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param learning_rate: the learning rate for training
    :param batch_size: the batch size for training and sampling
    :param diffusion_steps: the diffusion timesteps for the forward diffusion process
    :param epochs: the training epochs
    :param mlp_layers: number of neurons per layer
    :param activation: activation function - "lrelu", "relu", "tanh" or "sigmoid"
    :param dim_t: dimensionality of the intermediate layer for connecting the embeddings.
    :param cat_emb_dim: dimension of categorical embeddings
    :param diff_beta_start_end: diffusion start and end betas
    :param scheduler: diffusion scheduler - "linear" or "quad"
    :param epsilon: the privacy budget of the differential privacy.
        One can specify how the budget is split between pre-processing and fitting the model.
        For example, epsilon = {"preprocessing": 0.1, "fitting": 0.9}.
        If a single float number is provide, half the budget is allocated for pre-processing
        and half for fitting.
    :param delta: target delta to be achieved for fitting (for differentially private model)
    :param max_grad_norm: the maximum norm of the per-sample gradients.
        Any gradient with norm higher than this will be clipped to this value. (for differentially private model)
    :param preprocess_metadata: specify the range (minimum and maximum) for all numerical columns
        and the distinct categories for categorical columns. This ensures that no further privacy leakage
        is happening. For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        If not specified, they will be estimated from the real data and a warning will be raised
        (for differentially private model)
    """

    name = "FinDiff"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        learning_rate: int = 1e-4,
        batch_size: int = 512,
        diffusion_steps: int = 500,
        epochs: int = 500,
        mpl_layers: list[int] = [1024, 1024, 1024, 1024],
        activation: str = "lrelu",
        dim_t: int = 64,
        cat_emb_dim: int = 2,
        diff_beta_start_end: list[float] = [1e-4, 0.02],
        scheduler: str = "linear",
        epsilon: Union[float, Dict[str, float]] = None,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        preprocess_metadata: Dict[str, dict] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        bounds = preprocess_metadata

        # assert not any(
        #    df.columns.str.contains("_")
        # ), "Please remove the '_' from column names for correct inverse decoding"

        # set seeds
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

        # set the device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ).type

        # Determine categorical and numerical attributes
        self.cat_attrs = metadata["categorical"]
        self.num_attrs = metadata["continuous"]

        self._df = self._df.copy()
        self.df_original = (
            self._df.copy()
        )  # self._df will be altered, so keep an original copy

        self.generator_filepath = generator_filepath
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.diffusion_steps = diffusion_steps
        self.epochs = epochs
        self.mpl_layers = mpl_layers
        self.activation = activation
        self.dim_t = dim_t
        self.cat_emb_dim = cat_emb_dim
        self.diff_beta_start_end = diff_beta_start_end
        self.scheduler = scheduler

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.bounds = bounds

        self._contains_num, self._contains_cat = False, False

        if len(self._metadata["continuous"]) > 0:
            self._contains_num = True
        if len(self._metadata["categorical"]) > 0:
            self._contains_cat = True

        if epsilon == {"preprocessing": None, "fitting": None}:
            self.epsilon = None

        # Initiate privacy budget
        if epsilon is not None:
            if not isinstance(epsilon, dict):
                if self.bounds is not None:
                    self.epsilon = {
                        "preprocessing": epsilon / 2,
                        "fitting": epsilon / 2,
                    }
                else:
                    self.epsilon = {"preprocessing": 0, "fitting": epsilon}
            self.eps_spent_preprocessing = 0

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # Non-DP pre-processing or DP pre-processing but with no bounds provided
        if (
            (self.epsilon is None)
            or (self.epsilon.get("preprocessing", None) is None)
            or (
                self.epsilon.get("preprocessing", None) is not None
                and self.bounds is None
            )
        ):
            if isinstance(self.epsilon, dict) and (
                (self.epsilon.get("preprocessing", None) is not None)
                and self.bounds is None
            ):  # Privacy leakage warning
                warnings.warn(
                    f"No bounds are specified for all the columns. DP is not guaranteed as it is not used in data preprocessing.",
                    PrivacyLeakWarning,
                )

            self._df[self.cat_attrs] = self._df[self.cat_attrs].astype("category")
            # Create a dictionary with all the unique categories (all columns in one dict)
            # This will then be used to map back the created names to the original names of the categories
            self._cat_dict = {}

            for cat_attr in self.cat_attrs:
                # Add unique values and their mapping to the mapping dictionary created above
                unique_values = self._df[cat_attr].unique()
                unique_dict = {f"{cat_attr}_{value}": value for value in unique_values}
                self._cat_dict.update(unique_dict)

                # add col name to every categorical entry to make them distinguishable for embedding
                self._df[cat_attr] = cat_attr + "_" + self._df[cat_attr].astype("str")

            # Reorder cols
            self._df = self._df[[*self.cat_attrs, *self.num_attrs]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quantile transformation for numerical vars
                self.num_scaler = QuantileTransformer(
                    output_distribution="normal", random_state=self._random_state
                )
                self.num_scaler.fit(self._df[self.num_attrs].values)
                train_num_scaled = self.num_scaler.transform(
                    self._df[self.num_attrs].values
                )

            # Transform the categorical attributes into ordinal numbers
            vocabulary_classes = np.unique(
                self._df[self.cat_attrs]
            )  # get vocabulary of categorical attributes

            # determine number unique categorical tokens
            self.n_cat_tokens = len(vocabulary_classes)

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(vocabulary_classes)
            train_cat_scaled = self._df[self.cat_attrs].apply(
                self.label_encoder.transform
            )

            # collect unique values of each categorical attribute
            self.vocab_per_attr = {
                cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in self.cat_attrs
            }

            # Convert to tensor
            train_num_torch = torch.FloatTensor(train_num_scaled)
            train_cat_torch = torch.LongTensor(train_cat_scaled.values)

            train_set = TensorDataset(
                train_cat_torch,  # categorical attributes
                train_num_torch,  # numerical attributes
            )

            self.dataloader = DataLoader(
                dataset=train_set,
                batch_size=self.batch_size,
                num_workers=0,  # number of workers, useful for parallelization
                shuffle=True,
            )

        else:  # DP mode with (at least some) bounds provided
            self._df[self.cat_attrs] = self._df[self.cat_attrs].astype("category")
            # Create a dictionary with all the unique categories (all columns in one dict)
            # This will then be used to map back the created names to the original names of the categories
            self._cat_dict = {}

            for cat_attr in self.cat_attrs:
                # Add unique values and their mapping to the mapping dictionary created above
                unique_values = self._df[cat_attr].unique()
                unique_dict = {f"{cat_attr}_{value}": value for value in unique_values}
                self._cat_dict.update(unique_dict)

                # add col name to every categorical entry to make them distinguishable for embedding
                self._df[cat_attr] = cat_attr + "_" + self._df[cat_attr].astype("str")

            # Reorder cols
            self._df = self._df[[*self.cat_attrs, *self.num_attrs]]

            # Generate synthetic data for all the continuous variables.
            # The transformer is trained on the synthetic data to prevent privacy leakage.
            # Note the correlation between variables are not preserved, but this won't affect the data transformation.

            # Calculate the budget allocated to each variable
            if len(self._metadata["continuous"]) > 0:
                eps_per_var = self.epsilon["preprocessing"] / len(self.num_attrs)

                df_num_synth = pd.DataFrame(
                    index=range(len(self._df) * 100), columns=self.num_attrs
                )  # Generate 100x more data by default

                for col, series in self._df[self.num_attrs].items():
                    if col not in self.bounds:
                        warnings.warn(
                            f"Upper and lower bounds not specified for column '{col}', transformer will be trained on real data for this variable",
                            PrivacyLeakWarning,
                        )
                        df_num_synth[col][: len(self._df)] = self._df[col]
                    else:
                        self.eps_spent_preprocessing += eps_per_var
                        df_num_synth[col] = generate_synth_var(
                            var=self._df[col],
                            min=self.bounds[col]["min"],
                            max=self.bounds[col]["max"],
                            epsilon=eps_per_var,
                            n=len(self._df) * 100,  # Generate 100x more data by default
                            n_bins=100,  # 100 bins by default
                        )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Quantile transformation for numerical vars
                    self.num_scaler = QuantileTransformer(
                        output_distribution="normal", random_state=self._random_state
                    )
                    # Fit on synthetic data
                    self.num_scaler.fit(df_num_synth.values)
                    # Transform the real data
                    train_num_scaled = self.num_scaler.transform(
                        self._df[self.num_attrs].values
                    )

            # Transform the categorical attributes into ordinal numbers
            vocabulary_classes = []

            for col, series in self._df[self.cat_attrs].items():
                if col not in self.bounds:
                    warnings.warn(
                        f"List of categories not specified for column '{col}', categories will be extracted from real data for this variable",
                        PrivacyLeakWarning,
                    )
                    unique_value = list(np.unique(self._df[col]))
                else:
                    unique_value_raw = self.bounds[col]["categories"]
                    unique_value = [col + "_" + str(val) for val in unique_value_raw]

                vocabulary_classes += unique_value

            vocabulary_classes = np.sort(vocabulary_classes)

            # determine number unique categorical tokens
            self.n_cat_tokens = len(vocabulary_classes)

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(vocabulary_classes)
            train_cat_scaled = self._df[self.cat_attrs].apply(
                self.label_encoder.transform
            )

            # collect unique values of each categorical attribute
            self.vocab_per_attr = {
                cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in self.cat_attrs
            }

            # Convert to tensor
            if self._contains_num:
                train_num_torch = torch.FloatTensor(train_num_scaled)
            else:
                train_num_torch = None

            if self._contains_cat:
                train_cat_torch = torch.LongTensor(train_cat_scaled.values)
            else:
                train_cat_torch = None

            train_set = TensorDataset(
                *[
                    tens
                    for tens in [train_cat_torch, train_num_torch]
                    if tens is not None
                ]
            )

            self.dataloader = DataLoader(
                dataset=train_set,
                batch_size=self.batch_size,
                num_workers=0,  # number of workers, useful for parallelization
                shuffle=True,
            )

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Fit the diffusion model.

        :param save_path: the path to save the generator
        :return: *None*
        """

        if self.generator_filepath is None:
            # determine total categorical embedding dimension
            cat_dim = self.cat_emb_dim * len(self.cat_attrs)

            # determine total numerical embedding dimension
            num_dim = len(self.num_attrs)

            # determine total embedding dimension
            encoded_dim = cat_dim + num_dim

            self._gen = FinDiff(
                d_in=encoded_dim,
                hidden_layers=self.mpl_layers,
                activation=self.activation,
                dim_t=self.dim_t,
                contains_cat=self._contains_cat,
                contains_num=self._contains_num,
                n_cat_tokens=self.n_cat_tokens,
                n_cat_emb=self.cat_emb_dim,
                embedding=None,
                embedding_learned=False,
                n_classes=None,
                total_steps=self.diffusion_steps,
                beta_start=self.diff_beta_start_end[0],
                beta_end=self.diff_beta_start_end[1],
                scheduler=self.scheduler,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                epsilon=(
                    self.epsilon.get("fitting", None)
                    if isinstance(self.epsilon, dict)
                    else None
                ),
                delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                accountant="gdp",
                device=self.device,
            )

        # Deactivate the package prints while fitting the model
        with ustandard.HiddenPrints():
            if self.epsilon is None:
                self._gen.fit(self.dataloader, label=False)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._gen.fit_dp(self.dataloader, label=False)

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=FinDiffGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the parameters of FinDiff.

        :return: *None*
        """
        print("Generator: FinDiff")
        print("Model parameters: ", self._gen.params)

        if self.epsilon is not None:
            print(
                "Spent budget: ",
                {
                    "preprocessing": self.eps_spent_preprocessing,
                    "fitting": self._gen.params.get("eps_spent", None),
                },
            )

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using trained generator.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        kwargs = {
            "n_samples": num_samples,
            "label": None,
            "num_attrs": self.num_attrs,
            "cat_attrs": self.cat_attrs,
            "vocab_per_attr": self.vocab_per_attr,
            "label_encoder": self.label_encoder,
        }

        # Only add num_scaler if self.num_scaler exists
        if hasattr(self, "num_scaler"):
            kwargs["num_scaler"] = self.num_scaler
        else:
            kwargs["num_scaler"] = None

        with ustandard.HiddenPrints():
            samples = self._gen.sample(**kwargs)

        # Remove the prefix from the categorical variables
        for cat_attr in self.cat_attrs:
            # samples[cat_attr] = samples[cat_attr].str.split("_", n=1).str.get(-1)
            samples[cat_attr] = samples[cat_attr].replace(self._cat_dict)

        # Post-processing
        samples = transform_data(
            df_ref=self.df_original,
            df_to_trans=samples,
            cont_col=self._metadata["continuous"],
        )

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{FinDiffGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        return samples
