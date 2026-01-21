"""
Copyright 2024 - Business Source License 1.1 (https://github.com/sdv-dev/SDV)
The Licensed Work is (c) DataCebo, Inc.

You may make use of the Licensed Work, and derivatives of the Licensed
Work, provided that you do not use the Licensed Work, or derivatives of
the Licensed Work, for a Synthetic Data Creation Service.

You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://github.com/sdv-dev/SDV/blob/main/LICENSE

The original code was modified in the following ways to accommodate differential private training:
- Import of local modified TVAE
- Modification of the class methods to integrate DP training
"""

from clover.generators.external.ctgan.synthesizers.tvae import TVAE
from clover.generators.external.ctgan.synthesizers.ctgan import CTGAN

"""Wrapper around CTGAN model."""

# from ctgan import CTGAN

from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns


class CTGANSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``CTGAN`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {"categorical": None}

    def __init__(
        self,
        metadata,
        preprocess_metadata=None,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        epsilon=None,
        preprocess_epsilon_pp=None,
        delta=None,
        max_grad_norm=1,
        cuda=True,
    ):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.preprocess_metadata = preprocess_metadata
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.epsilon = epsilon
        self.preprocess_epsilon_pp = preprocess_epsilon_pp
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.cuda = cuda

        self._model_kwargs = {
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "log_frequency": log_frequency,
            "verbose": verbose,
            "epochs": epochs,
            "pac": pac,
            "epsilon": epsilon,
            "preprocess_epsilon_pp": preprocess_epsilon_pp,
            "delta": delta,
            "max_grad_norm": max_grad_norm,
            "cuda": cuda,
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data)
        self._model = CTGAN(**self._model_kwargs)

        if self._model.epsilon is not None:
            self._model.fit_dp(
                processed_data,
                discrete_columns=discrete_columns,
                preprocess_metadata=self.preprocess_metadata,
            )

        else:
            self._model.fit(
                processed_data,
                discrete_columns=discrete_columns,
            )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError(
            "CTGANSynthesizer doesn't support conditional sampling."
        )


class TVAESynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``TVAE`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        compress_dims (tuple or list of ints):
            Size of each hidden layer in the encoder. Defaults to (128, 128).
        decompress_dims (tuple or list of ints):
           Size of each hidden layer in the decoder. Defaults to (128, 128).
        l2scale (int):
            Regularization term. Defaults to 1e-5.
        batch_size (int):
            Number of data samples to process in each step.
        epochs (int):
            Number of training epochs. Defaults to 300.
        loss_factor (int):
            Multiplier for the reconstruction error. Defaults to 2.
        epsilon (float or None):
            Privacy parameter for differential privacy. Defaults to None.
        delta (float or None):
            Privacy parameter for differential privacy. Defaults to None.
        max_grad_norm (Union[float, List[float]]):
            The maximum norm of the per-sample gradients. Defaults to 1.
        max_physical_batch_size (int):
            Maximum number of samples processed at a time during training. Defaults to 126.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {"categorical": None}

    def __init__(
        self,
        metadata,
        preprocess_metadata=None,
        enforce_min_max_values=True,
        enforce_rounding=True,
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
        cuda=True,
    ):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        self.preprocess_metadata = preprocess_metadata
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.cuda = cuda
        self.epsilon = epsilon
        self.preprocess_epsilon_pp = preprocess_epsilon_pp
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size

        self._model_kwargs = {
            "embedding_dim": embedding_dim,
            "compress_dims": compress_dims,
            "decompress_dims": decompress_dims,
            "l2scale": l2scale,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss_factor": loss_factor,
            "epsilon": epsilon,
            "preprocess_epsilon_pp": preprocess_epsilon_pp,
            "delta": delta,
            "max_grad_norm": max_grad_norm,
            "max_physical_batch_size": max_physical_batch_size,
            "cuda": cuda,
        }

    def _fit(
        self,
        processed_data,
    ):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data)

        self._model = TVAE(**self._model_kwargs)

        if self._model.epsilon is not None:
            self._model.fit_dp(
                processed_data,
                preprocess_metadata=self.preprocess_metadata,
                discrete_columns=discrete_columns,
            )
        else:
            self._model.fit(
                processed_data,
                discrete_columns=discrete_columns,
            )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError(
            "TVAESynthesizer doesn't support conditional sampling."
        )
