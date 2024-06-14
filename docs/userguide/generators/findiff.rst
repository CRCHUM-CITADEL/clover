FinDiff
=======

Introduction
------------

Diffusion models, initially introduced by Ho et al., have achieved considerable success in the field of computer vision. Similar to Generative Adversarial Networks (GANs), they aim to approximate the probability distribution of a given domain and provide a means to sample from that distribution. These models follow a unique approach inspired by non-equilibrium thermodynamics. In the case of diffusion models, a random noise signal is progressively refined until it closely aligns with the desired data distribution. Consequently, diffusion models have emerged as a powerful technique for generating complex data. Sattarov et al. have further adapted the diffusion model framework for tabular data generation, demonstrating superior performance compared to VAE and GAN models in this context.


Algorithm
---------

The diffusion model comprises both a forward diffusion process and a reverse denoising diffusion process. These processes occur over predefined finite time steps. In the forward diffusion process, we initially sample real data from our data distribution. Noise is sampled at each subsequent step, and this noise is added to the data from the previous step. A neural network is trained to predict the added noise at each step. By following a well-behaved schedule for adding noise at each time step and using a sufficiently large number of steps, we eventually obtain pure noise. In the reverse denoising process, the data is gradually denoised, starting from pure noise and progressing toward realistic data.

When generating synthetic data with diffusion models, we begin by sampling pure noise and then progressively denoise it using the trained neural network, leveraging the conditional probability it has learned.

Generating tabular data with diffusion models presents challenges due to the inherent heterogeneity of such data. Each variable in tabular data may have a different nature. To address these issues, FinDiff converts categorical data into semantically rich continuous representations using embeddings. Additionally, it integrates time and label embeddings into the model input, enhancing the overall input quality.

Below is the pseudo code for generating synthetic data with diffusion model.


.. code-block:: none

    # Step 1: model training
    procedure TRAINING(X_0, θ, T):
        for step t ∈ {1,...,T} do
            z_t = sample_noise(t)
            X_t = add_noise_to_data(X_t-1, z_t)
            M_θ.fit(X_t, z_t)
        return M_θ

    # Step 2: data sampling
    procedure SAMPLING(M_θ, T):
        X'_T= sample_pure_noise()
        for step t = T down to 1 do
            z'_t = M_θ.pred(X'_t)
            X'_t-1= denoise_data(X'_t, z'_t)
        return X'_0

In this pseudocode:

- X_0 is the training data.
- θ is the parameters of the model M_θ.
- T is the total number of steps.


Clover implementation
---------------------

.. code-block:: python

    class FinDiffGenerator(Generator):
        """
        Wrapper of the tabular diffusion models FinDiff https://github.com/sattarov/FinDiff.
        The original model was modified to implement differential privacy.

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
        :param bounds: specify the range (minimum and maximum) for all numerical columns
            and the distinct categories for categorical columns. This ensures that no further privacy leakage
            is happening. For example, bounds = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
            If not specified, they will be estimated from the real data and a warning will be raised
            (for differentially private model)
        """


Steps include:

#.
   Preparing the parameters to train the generator.
#.
   Train the generator and save it.
#.
   Generate samples using trained model.


References
----------

- `Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851. <https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf>`_

- `Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., & Welling, M. (2021). Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34, 12454-12465. <https://proceedings.neurips.cc/paper/2021/file/67d96d458abdef21792e6d8e590244e7-Paper.pdf>`_

- `Sattarov, T., Schreyer, M., & Borth, D. (2023). Findiff: Diffusion models for financial tabular data generation. In Proceedings of the Fourth ACM International Conference on AI in Finance 64-72. <https://dl.acm.org/doi/abs/10.1145/3604237.3626876>`_

- https://github.com/CRCHUM-CITADEL/clover/blob/main/generators/findiff_generator.py

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/synthetic_data_generation.ipynb
