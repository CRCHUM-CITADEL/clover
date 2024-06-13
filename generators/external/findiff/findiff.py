import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from .MLPSynthesizer import MLPSynthesizer
from .BaseDiffuser import BaseDiffuser
from .findiff_modules import train, train_dp, generate_samples, decode_sample


class FinDiff:
    """FinDiff synthetic data generator

    :param d_in: dimensionality of the input data
    :param hidden_layers: list of the neurons in every hidden layer
    :param activation: activation function. Defaults to 'lrelu'
    :param dim_t: dimensionality of the intermediate layer for connecting the embeddings. Defaults to 64.
    :param n_cat_tokens: number of total categorical tokens. Defaults to None
    :param n_cat_emb: dim of categorical embeddings. Defaults to None
    :param embedding: provide if learned embeddings are given. Defaults to None
    :param embedding_learned: flag whether the embeddings need to be learned. Defaults to True
    :param n_classes: total number of classes, if conditional sampling is required. Defaults to None
    :param total_steps: total diffusion steps. Defaults to 1000
    :param beta_start: beta start value. Defaults to 1e-4
    :param beta_end: beta end value. Defaults to 0.02
    :param scheduler: diffusion scheduler type
    :param learning_rate: learning rate
    :param epochs: training epochs
    :param epsilon: the privacy budget,
    :param delta: target delta to be achieved for fitting (for differentially private model)
    :param max_grad_norm: he maximum norm of the per-sample gradients (for differentially private model)
    :param accountant: accounting mechanism, i.e. rdp for RDP accountant and gdp for Gaussian accountant (for differentially private model)
    :param device: either cpu or cuda
    """

    def __init__(
        self,
        d_in: int,
        hidden_layers: list,
        activation: str = "lrelu",
        dim_t: int = 64,
        n_cat_tokens: int = None,
        n_cat_emb: int = None,
        embedding: torch.tensor = None,
        embedding_learned: bool = True,
        n_classes: int = None,
        total_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        scheduler: str = "linear",
        learning_rate: float = None,
        epochs: int = None,
        epsilon: float = None,
        delta: float = None,
        max_grad_norm: float = None,
        accountant: str = "gdp",
        device: str = None,
    ):
        self.params = {}
        self.params["d_in"] = d_in
        self.params["hidden_layers"] = hidden_layers
        self.params["activation"] = activation
        self.params["dim_t"] = dim_t
        self.params["n_cat_tokens"] = n_cat_tokens
        self.params["n_cat_emb"] = n_cat_emb
        self.params["embedding"] = embedding
        self.params["embedding_learned"] = embedding_learned
        self.params["n_classes"] = n_classes
        self.params["total_steps"] = total_steps
        self.params["beta_start"] = beta_start
        self.params["beta_end"] = beta_end
        self.params["scheduler"] = scheduler
        self.params["learning_rate"] = learning_rate
        self.params["epochs"] = epochs
        self.params["epsilon"] = epsilon
        self.params["delta"] = delta
        self.params["max_grad_norm"] = max_grad_norm
        self.params["accountant"] = accountant
        self.params["device"] = device

        # initialize the FinDiff synthesizer model
        self.synthesizer_model = MLPSynthesizer(
            d_in=d_in,
            hidden_layers=hidden_layers,
            activation=activation,
            dim_t=dim_t,
            n_cat_tokens=n_cat_tokens,
            n_cat_emb=n_cat_emb,
            embedding=embedding,
            embedding_learned=embedding_learned,
            n_classes=n_classes,
        )

        # initialize the FinDiff base diffuser model
        self.diffuser_model = BaseDiffuser(
            total_steps=total_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            scheduler=scheduler,
            device=device,
        )

    def fit(self, train_data: DataLoader, label: bool) -> dict:
        """Fit the diffusion model

        :param train_data: training data
        :param label: if label is contained in the dataloader
        :return: training losses for each epoch
        """
        # determine synthesizer model parameters; filter out the parameters to be learned
        parameters = filter(
            lambda p: p.requires_grad, self.synthesizer_model.parameters()
        )

        # init Adam optimizer
        optimizer = optim.Adam(parameters, lr=self.params["learning_rate"])

        # int mean-squared-error loss
        loss_fnc = nn.MSELoss()

        losses_dict, self.synthesizer_model, self.diffuser_model = train(
            dataloader=train_data,
            label=label,
            synthesizer=self.synthesizer_model,
            diffuser=self.diffuser_model,
            loss_fnc=loss_fnc,
            optimizer=optimizer,
            epochs=self.params["epochs"],
            device=self.params["device"],
        )

        return losses_dict

    def fit_dp(self, train_data: DataLoader, label: bool) -> tuple:
        """Fit the differentially private diffusion model

        :param train_data: training data
        :param label: if label is contained in the dataloader
        :return: training losses for each epoch and the spent budget
        """
        # determine synthesizer model parameters; filter out the parameters to be learned
        parameters = filter(
            lambda p: p.requires_grad, self.synthesizer_model.parameters()
        )

        # init Adam optimizer
        optimizer = optim.Adam(parameters, lr=self.params["learning_rate"])

        # int mean-squared-error loss
        loss_fnc = nn.MSELoss()

        (
            losses_dict,
            eps_spent,
            self.synthesizer_model,
            self.diffuser_model,
        ) = train_dp(
            dataloader=train_data,
            label=label,
            synthesizer=self.synthesizer_model,
            diffuser=self.diffuser_model,
            loss_fnc=loss_fnc,
            optimizer=optimizer,
            epochs=self.params["epochs"],
            epsilon=self.params["epsilon"],
            delta=self.params["delta"],
            max_grad_norm=self.params["max_grad_norm"],
            accountant=self.params["accountant"],
            device=self.params["device"],
        )

        self.params["eps_spent"] = eps_spent

        return losses_dict, eps_spent

    def sample(
        self,
        n_samples: int,
        label: torch.tensor,
        num_attrs: list,
        cat_attrs: list,
        vocab_per_attr: dict,
        num_scaler: QuantileTransformer,
        label_encoder: LabelEncoder,
    ) -> pd.DataFrame:
        """Generate synthetic data with trained model

        :param n_samples: number of samples to sample
        :param label: list of labels for conditional sampling
        :param num_attrs: numeric attributes
        :param cat_attrs: categorical attributes
        :param vocab_per_attr: vocabulary of distinct values in attribute
        :param num_scaler: numeric scaler from sklearn
        :param label_encoder: categorical encoder

        :return: synthetic data
        """
        raw_samples = generate_samples(
            synthesizer=self.synthesizer_model,
            diffuser=self.diffuser_model,
            encoded_dim=self.params["d_in"],
            last_diff_step=self.params["total_steps"],
            n_samples=n_samples,
            label=label,
            device=self.params["device"],
        )

        n_cat_dim = self.params["n_cat_emb"] * len(cat_attrs)

        samples = decode_sample(
            sample=raw_samples,
            cat_dim=n_cat_dim,
            n_cat_emb=self.params["n_cat_emb"],
            num_attrs=num_attrs,
            cat_attrs=cat_attrs,
            num_scaler=num_scaler,
            vocab_per_attr=vocab_per_attr,
            label_encoder=label_encoder,
            synthesizer=self.synthesizer_model,
        )

        return samples
