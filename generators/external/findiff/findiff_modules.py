"""
The following modifications were made to the file:
    - The full training loop was added
    - Train function was modified to train for n epochs
    - Input and target was reversed in the loss function
    - Steps sampling was fixed for reverse diffusion process
    - Device parameter was added to generate_samples function
    - Issue was fix in generate_samples function when label is None
    - Differential privacy was implemented
    - Learning rate scheduler was implemented in training
"""
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from opacus import PrivacyEngine

from .MLPSynthesizer import get_embeddings, embed_categorical


def train(
    dataloader,
    label,
    synthesizer,
    diffuser,
    loss_fnc,
    optimizer,
    epochs,
    device,
):
    """Training model

    Args:
        dataloader (_type_): torch Dataloader
        label (bool): if the dataloader contains label
        synthesizer (_type_): model synthesizer
        diffuser (_type_): diffuser model
        loss_fnc (_type_): loss function
        optimizer (_type_): optimizer
        epochs (int): training epochs
        device (str): cpu or gpu

    Returns:
        tuple: losses, synthesizer and diffuser
    """
    # init collection of training epoch losses
    train_epoch_losses = []

    # set network in training mode
    synthesizer.train()

    # move to the device
    synthesizer = synthesizer.to(device)

    # init learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # init the training progress bar
    pbar = tqdm(iterable=range(epochs), position=0, leave=True)

    # iterate over training epochs
    for epoch in pbar:
        # init epoch training batch losses
        batch_losses = []

        # iterate over distinct mini-batches
        for elements in dataloader:
            if label:
                (batch_cat, batch_num, batch_y) = elements
            else:
                (batch_cat, batch_num) = elements
                batch_y = None

            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            if batch_y is not None:
                batch_y = batch_y.to(device)

            # sample timestamps t for each observation in a minibatch
            timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

            # get cat embeddings
            batch_cat_emb = embed_categorical(
                embedding=synthesizer.embedding, x_cat=batch_cat
            )

            # concat cat & num
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

            # add noise
            batch_noise_t, noise_t = diffuser.add_gauss_noise(
                x_num=batch_cat_num, t=timesteps
            )

            # conduct forward encoder/decoder pass
            predicted_noise = synthesizer(
                x=batch_noise_t, timesteps=timesteps, label=batch_y
            )

            # compute batch loss
            train_losses = loss_fnc(
                input=predicted_noise,
                target=noise_t,
            )

            # reset encoder and decoder gradients
            optimizer.zero_grad()

            # run error back-propagation
            train_losses.backward()

            # optimize model parameters
            optimizer.step()

            # collect training batch error losses
            batch_losses.append(train_losses.detach().cpu().numpy())

        # average of batch errors
        batch_losses_mean = np.mean(np.array(batch_losses))

        # update learning rate according to the scheduler
        scheduler.step()

        # collect mean training epoch loss
        train_epoch_losses.append(batch_losses_mean)

        # prepare and set training epoch progress bar update
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_description(
            "[LOG {}] epoch: {}, train-loss: {}".format(
                str(now), str(epoch).zfill(4), str(batch_losses_mean)
            )
        )

    return {"losses": train_epoch_losses}, synthesizer, diffuser


def train_dp(
    dataloader,
    label,
    synthesizer,
    diffuser,
    loss_fnc,
    optimizer,
    epochs,
    epsilon,
    delta,
    max_grad_norm,
    accountant,
    device,
):
    """Training differentially private model

    Args:
        dataloader (_type_): torch Dataloader
        label (bool): if the dataloader contains label
        synthesizer (_type_): model synthesizer
        diffuser (_type_): diffuser model
        loss_fnc (_type_): loss function
        optimizer (_type_): optimizer
        epochs (int): training epochs
        epsilon (float): the privacy budget
        delta (float): target delta to be achieved for fitting
        max_grad_norm (float): he maximum norm of the per-sample gradients
        accountant (str): accounting mechanism, i.e. rdp for RDP accountant and gdp for Gaussian accountant
        device (str): cpu or gpu

    Returns:
        tuple: losses, spent budget, synthesizer, and diffuser
    """

    # init collection of training epoch losses
    train_epoch_losses = []

    privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=False)
    (
        synthesizer,
        optimizer,
        dataloader,
    ) = privacy_engine.make_private_with_epsilon(
        module=synthesizer,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,
    )

    # init learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # move to the device
    synthesizer = synthesizer.to(device)

    # set network in training mode
    synthesizer.train()

    # initiate spent budget
    eps_spent = 0

    # init the training progress bar
    pbar = tqdm(iterable=range(epochs), position=0, leave=True)

    # iterate over training epochs
    for epoch in pbar:
        # init epoch training batch losses
        batch_losses = []

        # iterate over distinct mini-batches
        for elements in dataloader:
            if label:
                (batch_cat, batch_num, batch_y) = elements
            else:
                (batch_cat, batch_num) = elements
                batch_y = None

            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            if batch_y is not None:
                batch_y = batch_y.to(device)

            # sample timestamps t for each observation in a minibatch
            timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

            # get cat embeddings
            batch_cat_emb = embed_categorical(
                embedding=synthesizer.embedding, x_cat=batch_cat
            )

            # concat cat & num
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

            # add noise
            batch_noise_t, noise_t = diffuser.add_gauss_noise(
                x_num=batch_cat_num, t=timesteps
            )

            # conduct forward encoder/decoder pass
            predicted_noise = synthesizer(
                x=batch_noise_t, timesteps=timesteps, label=batch_y
            )

            # compute batch loss
            train_losses = loss_fnc(
                input=predicted_noise,
                target=noise_t,
            )

            # reset encoder and decoder gradients
            optimizer.zero_grad()

            # run error back-propagation
            train_losses.backward()

            # optimize model parameters
            optimizer.step()

            # collect training batch error losses
            batch_losses.append(train_losses.detach().cpu().numpy())

            # fetch spent epsilon
            eps_spent = privacy_engine.get_epsilon(delta)

        # average of batch errors
        batch_losses_mean = np.mean(np.array(batch_losses))

        # update learning rate according to the scheduler
        scheduler.step()

        # collect mean training epoch loss
        train_epoch_losses.append(batch_losses_mean)

        # prepare and set training epoch progress bar update
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_description(
            "[LOG {}] epoch: {}, train-loss: {}".format(
                str(now), str(epoch).zfill(4), str(batch_losses_mean)
            )
        )

    return (
        {"losses": train_epoch_losses},
        eps_spent,
        synthesizer,
        diffuser,
    )


@torch.no_grad()
def generate_samples(
    synthesizer,
    diffuser,
    encoded_dim,
    last_diff_step,
    n_samples=None,
    label=None,
    device="cpu",
):
    """Generation of samples.
        For unconditional sampling use n_samples, for conditional sampling provide label.

    Args:
        synthesizer (_type_): synthesizer model
        diffuser (_type_): diffuzer model
        encoded_dim (int): transformed data dimension
        last_diff_step (int): total number of diffusion steps
        n_samples (int, optional): number of samples to sample. Defaults to None.
        label (tensor, optional): list of labels for conditional sampling. Defaults to None.
        device(str): cpu or gpu

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    if (n_samples is None) and (label is None):
        raise Exception("either n_samples or label needs to be given")

    if label is not None:
        n_samples = len(label)
        label = label.to(device)

    z_norm = torch.randn((n_samples, encoded_dim), device=device)  # Sample pure noise

    pbar = tqdm(iterable=reversed(range(1, last_diff_step + 1)))
    for i in pbar:
        pbar.set_description(f"SAMPLING STEP: {i:4d}")

        # init diffusion timesteps; create tensor of shape (n_samples,) filled wth diffusion_step
        t = torch.full((n_samples,), i, dtype=torch.long, device=device)

        model_out = synthesizer(z_norm.float(), t, label)

        z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

    return z_norm


def decode_sample(
    sample,
    cat_dim,
    n_cat_emb,
    num_attrs,
    cat_attrs,
    num_scaler,
    vocab_per_attr,
    label_encoder,
    synthesizer,
):
    """Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
        Used once synthetic data is generated.

    Args:
        sample (tensor): input samples for decoding
        cat_dim (int): categorical dimension
        n_cat_emb (int): size of categorical embeddings
        num_attrs (list): numeric attributes
        cat_attrs (list): categorical attributes
        num_scaler (_type_): numeric scaler from sklearn
        vocab_per_attr (dict): vocabulary of distinct values in attribute
        label_encoder (_type_): categorical encoder
        synthesizer (_type_): model synthesizer

    Returns:
        pandas DataFrame: decoded dataframe
    """

    # split sample into numeric and categorical parts
    sample = sample.cpu().numpy()
    sample_num = sample[:, cat_dim:]
    sample_cat = sample[:, :cat_dim]

    # denormalize numeric attributes
    z_norm_upscaled = num_scaler.inverse_transform(sample_num)
    z_norm_df = pd.DataFrame(z_norm_upscaled, columns=num_attrs)

    # get embedding lookup matrix
    embedding_lookup = get_embeddings(embedding=synthesizer.embedding).cpu()
    # reshape back to batch_size * n_dim_cat * cat_emb_dim
    sample_cat = sample_cat.reshape(-1, len(cat_attrs), n_cat_emb)
    # compute pairwise distances; shape = (# of sample, # of value in lookup, # of attributes)
    distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(sample_cat))
    # get the closest distance based on the embeddings that belong to a column category
    z_cat_df = pd.DataFrame(index=range(len(sample_cat)), columns=cat_attrs)
    nearest_dist_df = pd.DataFrame(index=range(len(sample_cat)), columns=cat_attrs)
    for attr_idx, attr_name in enumerate(cat_attrs):
        attr_emb_idx = list(vocab_per_attr[attr_name])  # in ascending order
        attr_distances = distances[:, attr_emb_idx, attr_idx]
        # nearest_idx = torch.argmin(attr_distances, dim=1).cpu().numpy()
        nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
        nearest_idx = nearest_idx.cpu().numpy()

        z_cat_df[attr_name] = np.array(attr_emb_idx)[
            nearest_idx
        ]  # need to map emb indices back to column indices
        nearest_dist_df[attr_name] = nearest_values.cpu().numpy()

    z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)
    sample_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

    return sample_decoded
