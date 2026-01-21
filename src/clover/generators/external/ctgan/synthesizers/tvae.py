"""
Copyright 2024 - Business Source License 1.1 (https://github.com/sdv-dev/CTGAN)
The Licensed Work is (c) DataCebo, Inc.

You may make use of the Licensed Work, and derivatives of the Licensed
Work, provided that you do not use the Licensed Work, or derivatives of
the Licensed Work, for a Synthetic Data Creation Service.

You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://github.com/sdv-dev/CTGAN/blob/main/LICENSE

The original code was modified in the following ways to accommodate differential private training:
- Import of opacus, the pytorch DP framework
- Modification of the structure of the model (Addition of the VAE module comprising the Encoder and Decoder
  modules)
- Addition of the function fit_dp which trains the model with differential privacy
"""

"""TVAE module."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from clover.utils.preprocessing import generate_continuous_dp


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`.
        DP modif: do not return self.sigma.
        It will be in the forward function of the VAE module.
        """
        return self.seq(input_)  # , self.sigma


class VAE(Module):
    """
    Variational Autoencoder (VAE) for generative modeling.
    Added to accommodate the DP version of TVAE with Opacus.

    Args:
        embedding_dim (int):
            Size of the latent embedding space.
        compress_dims (list or tuple of ints):
            Sizes of the hidden layers in the encoder.
        decompress_dims (list or tuple of ints):
            Sizes of the hidden layers in the decoder.
        data_dim (int):
            Dimensions of the input data.

    Attributes:
        encoder (Encoder):
            Encoder module for mapping input data to latent embeddings.
        decoder (Decoder):
            Decoder module for generating reconstructed data from latent embeddings.
    """

    def __init__(self, embedding_dim, compress_dims, decompress_dims, data_dim):
        super().__init__()
        self.encoder = Encoder(
            data_dim=data_dim, compress_dims=compress_dims, embedding_dim=embedding_dim
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            decompress_dims=decompress_dims,
            data_dim=data_dim,
        )

    def forward(self, _input):
        """
        Forward pass through the VAE.
            Args:
                _input (torch.Tensor): Input data.
            Returns:
                tuple: Tuple containing:
                    - mu (torch.Tensor): Mean of the latent embeddings.
                    - std (torch.Tensor): Standard deviation of the latent embeddings.
                    - logvar (torch.Tensor): Log variance of the latent embeddings.
                    - rec (torch.Tensor): Reconstructed data.
                    - sigmas (torch.Tensor): Standard deviations for each feature in the reconstructed data.
        """
        mu, std, logvar = self.encoder(_input)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        rec = self.decoder(emb)
        sigmas = self.decoder.sigma
        return mu, std, logvar, rec, sigmas


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed],
                        torch.argmax(x[:, st:ed], dim=-1),
                        reduction="sum",
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        epsilon=None,
        preprocess_epsilon_pp=None,
        delta=None,
        max_grad_norm=1,
        max_physical_batch_size=126,
        verbose=1,
        cuda=True,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size
        self.preprocess_epsilon_pp = preprocess_epsilon_pp

        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(
            torch.from_numpy(train_data.astype("float32")).to(self._device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        data_dim = self.transformer.output_dimensions

        # Code modified below to integrate the VAE module
        self.vae = VAE(
            data_dim=data_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            embedding_dim=self.embedding_dim,
        )

        self.vae.encoder = self.vae.encoder.to(self._device)
        self.vae.decoder = self.vae.decoder.to(self._device)

        self.optimizerAE = Adam(self.vae.parameters(), weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                self.optimizerAE.zero_grad()
                real = data[0].to(self._device)
                # Code modified below to integrate the VAE module
                mu, std, logvar, rec, sigmas = self.vae(real)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                self.optimizerAE.step()
                self.vae.decoder.sigma.data.clamp_(0.01, 1.0)

    def fit_dp(
        self,
        train_data,
        discrete_columns=(),
        preprocess_metadata=None,
    ):
        """
        Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            preprocess_metadata (dict of dict):
                Dictionary containing the information related to preprocessing of the columns.
                Minimum, maximum, number of bins and decimal places for continuous variables.
                List of expected categories for discrete variables.
            epsilon (float):
                Target privacy parameter (epsilon) for differential privacy.
            delta (float):
                Target privacy parameter (delta) for differential privacy.
            max_grad_norm (float):
                Maximum gradient norm for gradient clipping during training.
            max_physical_batch_size (int):
                Maximum batch size for privacy accounting.
            discrete_columns (list-like, optional):
                List of discrete columns to be used to generate the Conditional Vector.
                If ``train_data`` is a Numpy array, this list should contain the integer indices of the columns.
                Otherwise, if it is a ``pandas.DataFrame``, this list should contain the column names.
            verbose (bool):
                Whether to print the epochs, loss and epsilon values during training.
        """

        if self.preprocess_epsilon_pp is None:
            self.preprocess_epsilon_pp = 0.5
            warnings.warn(
                "Half of the privacy budget will be used for preprocessing purposes."
            )

        continuous_columns = [
            col for col in train_data.columns if col not in discrete_columns
        ]

        if self.preprocess_epsilon_pp == 0:
            warnings.warn(
                "preprocess_epsilon_pp was set to 0. No privacy budget will be dedicated to preprocessing. "
                "It will be based on the properties of the real data, which may not satisfy differential privacy."
            )

        if preprocess_metadata is None:
            warnings.warn(
                "Metadata for preprocessing was not provided. It will be generated based on the properties "
                "of the real data, which may not satisfy differential privacy."
            )
            preprocess_metadata = {}

        for col in train_data.columns:
            if col not in preprocess_metadata.keys() and col in continuous_columns:
                warnings.warn(
                    f"Metadata for preprocessing {col} was not provided. It will be generated based on the properties "
                    "of the real data, which may not satisfy differential privacy."
                )
                preprocess_metadata[col] = {
                    "min_val": np.min(train_data[col]),
                    "max_val": np.max(train_data[col]),
                }
            elif col in preprocess_metadata.keys() and col in continuous_columns:
                assert (
                    "min_val" in preprocess_metadata[col].keys()
                    and "max_val" in preprocess_metadata[col].keys()
                ), (
                    f"Both the minimum and maximum values of {col} need to be specified in the metadata for "
                    f"preprocessing."
                )
            if col not in preprocess_metadata.keys() and col in discrete_columns:
                preprocess_metadata[col] = train_data[col].unique()
            elif col in preprocess_metadata.keys() and col in discrete_columns:
                assert isinstance(preprocess_metadata[col], list) or isinstance(
                    preprocess_metadata[col], np.ndarray
                ), (
                    f"{col} is a categorical variable. The metadata for preprocessing should be a list of unique "
                    f"categories."
                )

        preprocess_data = train_data.copy()
        preprocess_data = pd.concat([preprocess_data] * 100, ignore_index=True)

        # CHECK IF EPSILON NEEDS TO BE DIVIDED BY THE NUMBER OF COLUMNS
        if self.preprocess_epsilon_pp > 0:
            for col in continuous_columns:
                preprocess_data[col] = generate_continuous_dp(
                    df=preprocess_data,
                    col=col,
                    epsilon=self.epsilon
                    * self.preprocess_epsilon_pp
                    / len(continuous_columns),
                    sensitivity=1,
                    **preprocess_metadata[col],
                )

        categories_dict = {col: preprocess_metadata[col] for col in discrete_columns}

        self.transformer = DataTransformer()
        self.transformer.fit(
            preprocess_data, discrete_columns, categories_dict=categories_dict
        )
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(
            torch.from_numpy(train_data.astype("float32")).to(self._device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        data_dim = self.transformer.output_dimensions
        self.vae = VAE(
            data_dim=data_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            embedding_dim=self.embedding_dim,
        )

        self.vae.encoder = self.vae.encoder.to(self._device)
        self.vae.decoder = self.vae.decoder.to(self._device)

        self.optimizerAE = Adam(self.vae.parameters(), weight_decay=self.l2scale)

        privacy_engine = PrivacyEngine(accountant="rdp")
        (
            self.vae,
            self.optimizerAE,
            loader,
        ) = privacy_engine.make_private_with_epsilon(
            module=self.vae,
            optimizer=self.optimizerAE,
            data_loader=loader,
            target_delta=self.delta,
            target_epsilon=self.epsilon,
            max_grad_norm=self.max_grad_norm,
            epochs=self.epochs,
        )

        self.loss_values = pd.DataFrame(columns=["Epoch", "Batch", "Loss", "Epsilon"])

        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = "Loss: {loss:.3f}, Epsilon: {epsilon:.3f}"
            iterator.set_description(iterator_description.format(loss=0, epsilon=0))

        for i in iterator:
            loss_values = []
            epsilon_values = []
            batch = []

            with BatchMemoryManager(
                data_loader=loader,
                max_physical_batch_size=self.max_physical_batch_size,
                optimizer=self.optimizerAE,
            ) as memory_safe_data_loader:
                for id_, data in enumerate(memory_safe_data_loader):
                    self.optimizerAE.zero_grad()
                    real = data[0].to(self._device)
                    mu, std, logvar, rec, sigmas = self.vae(real)

                    loss_1, loss_2 = _loss_function(
                        rec,
                        real,
                        sigmas,
                        mu,
                        logvar,
                        self.transformer.output_info_list,
                        self.loss_factor,
                    )
                    loss = loss_1 + loss_2
                    loss.backward()
                    self.optimizerAE.step()
                    self.vae.decoder.sigma.data.clamp_(0.01, 1.0)

                    spent_epsilon = privacy_engine.get_epsilon(self.delta)

                    batch.append(id_)
                    loss_values.append(loss.detach().cpu().item())
                    epsilon_values.append(spent_epsilon)

                epoch_loss_df = pd.DataFrame(
                    {
                        "Epoch": [i] * len(batch),
                        "Batch": batch,
                        "Loss": loss_values,
                        "Epsilon": epsilon_values,
                    }
                )
                if not self.loss_values.empty:
                    self.loss_values = pd.concat(
                        [self.loss_values, epoch_loss_df]
                    ).reset_index(drop=True)
                else:
                    self.loss_values = epoch_loss_df

                if self.verbose:
                    iterator.set_description(
                        iterator_description.format(loss=loss, epsilon=spent_epsilon)
                    )

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.vae.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake = self.vae.decoder(noise)
            fake = torch.tanh(fake)
            sigmas = self.vae.decoder.sigma
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]

        self.vae.decoder.train()

        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
        self._device = device
        self.vae.decoder.to(self._device)
