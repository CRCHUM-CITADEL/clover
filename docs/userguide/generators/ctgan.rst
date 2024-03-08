Conditional Tabular GAN (CTGAN)
===============================

Introduction
------------

Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante and Kalyan Veeramachaneni introduced the conditional tabular GAN,
CTGAN, in 2019 (Xu et al., 2019). It aimed to solve what they identified as challenges in the existing models, namely
the ability to handle mixed data types, non-Gaussian distributions, multimodal distributions, highly imbalanced
categorical columns and sparse one-hot encoded vectors. The authors’ answer is manifold.

First, architectural changes are made to the vanilla GAN with the addition of the “pac” parameter from PacGAN
(Lin et al., 2018) and the adoption of
the Wassertein loss with gradient penalty from WGAN-GP (Gulrajani et al., 2017). The “pac” parameter relates to
the “packing” technique, which modifies the discriminator mechanism of the GAN to make decisions based on a group
of samples from the same class rather than individual observations. The packing technique addresses the issue of mode
collapse. In CTGAN’s implementation, the authors use a value of 10 samples in each pac.

The Wassertein GAN was first introduced by Arjovsky et al., and seeks to solve the issue of mode collapse and training
instability (Arjovsky et al., 2017). To do so, the authors replace the binary classification objective of the
discriminator by a measure of distance between the real and generated distributions, the Wassertein distance, also
called the Earth Mover’s (EM) distance. This distance allows the critic to reach optimality. The loss function needs
to have a maximum gradient for optimal training of the generator. Rather than weight clipping, Gulrajani et al.
propose a gradient penalty added to the loss function.

The second essential technique integrated to CTGAN is mode-specific normalization. This preprocessing step addresses
the non-Gaussian and multimodal distributions by estimating the number of modes and fitting a Gaussian mixture for
each continuous column. Each value in a continuous column is then represented as a one-hot vector indicating the mode
and a scalar indicating the value within the mode. A row becomes the concatenation of continuous and discrete
columns (Xu et al., 2019).

Finally, the authors add a conditional generator and training-by-sampling to address the imbalanced discrete
columns. A conditional vector “cond” is created to indicate the condition that a variable must take a certain
value, and integrated into the generator to penalize observations not complying with this condition.
Training-by-sampling samples the “cond” vector and the training data according to the log-frequency of each category
of the discrete columns. This combination allows the model to explore all possible values.

Below is the pseudo code of the CTGAN model.

.. code-block:: python

    # Step 1: data preprocessing
    function DataTransform(data, discrete_columns, continous_columns):
        cont = fit_mode_specific_GMM(data, continous_columns)
        disc = fit_one_hot_encoder(data, discrete_columns)
        transformed_data = concat(cont, disc)
        return transformed_data

    # Step 2: data sampling
    function DataSample(transformed_data, batch_size):
        cond = sample_original_condvec(transformed_data)
        sampled_data = sample_data(transformed_data, cond, log_frequency, batch_size)
        return sampled_data

    # Step 3: instantiate CTGAN
    function instantiate_CTGAN(params):
        discriminator = instantiate_Discriminator(pac)
        generator = instantiate_Generator()
        model = CTGAN(discriminator, generator)
        return model

    # Step 4: fit the model
    function fit_CTGAN(params):
        for i in epoch_iterator:
        for id_ in range(steps_per_epoch):
            for n in range(discriminator_steps):
                # Discriminator update
                sampled_data = DataSample(transformed_data, batch_size)
                update(discriminator, sampled_data)

            # Generator update
            sampled_data = DataSample(transformed_data, batch_size)
            update(generator, sampled_data)
        return model

Clover implementation
---------------------

.. code-block::

    class CTGANGenerator(Generator):
        """
        Wrapper of the GAN-based Deep Learning data synthesizer developed by Xu & al (Conditional Tabular GAN).
        https://github.com/sdv-dev

        See article for more information:
        Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
        Modeling tabular data using conditional GAN.
        Advances in Neural Information Processing Systems, 32.
        https://arxiv.org/abs/1907.00503

        :cvar name: the name of the generator
        :vartype name: str

        :param df: the data to synthesize
        :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
        :param random_state: for reproducibility purposes
        :param generator_filepath: the path of the generator to sample from if it exists
        :param discriminator_steps: the number of discriminator updates to do for each generator update.
        :param epochs: the number of training epochs.
        :param batch_size: the batch size for training.
        """

Note: CTGAN integrates packing and the Wassertain loss. Both address mode collapse. However, the authors of CTGAN
performed an ablation study and concluded that PacGAN might not be useful when WGAN-GP was used. This could be
explained by the Wassertein loss preventing mode collapse and thus rendering PacGAN redundant. Therefore, rather
than the default parameter of m=10 in CTGAN’s original paper, the default in the Clover library is m=1.

References
-----------
.. container:: references csl-bib-body hanging-indent
  :name: refs


  .. container:: csl-entry
     :name: ref-arjovsky_wasserstein_2017


     Arjovsky, Martin, Soumith Chintala, and Léon Bottou. 2017.
     “Wasserstein GAN.” arXiv. http://arxiv.org/abs/1701.07875.


  .. container:: csl-entry
     :name: ref-gulrajani_improved_2017


     Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin,
     and Aaron Courville. 2017. “Improved Training of Wasserstein
     GANs.” arXiv. http://arxiv.org/abs/1704.00028.


  .. container:: csl-entry
     :name: ref-lin_pacgan_2018


     Lin, Zinan, Ashish Khetan, Giulia Fanti, and Sewoong Oh. 2018.
     “PacGAN: The Power of Two Samples in Generative Adversarial
     Networks.” arXiv. http://arxiv.org/abs/1712.04086.


  .. container:: csl-entry
     :name: ref-xu_modeling_2019


     Xu, Lei, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan
     Veeramachaneni. 2019. “Modeling Tabular Data Using Conditional
     GAN.” arXiv. http://arxiv.org/abs/1907.00503.






