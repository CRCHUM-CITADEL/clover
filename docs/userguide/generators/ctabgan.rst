Conditional Table GAN (CTAB-GAN)
===============================

Introduction
------------

CTAB-GAN+ was proposed by Zilong Zhao, Aditya Kunar, Robert Birke and Lydia Y. Chen in 2022. It aimed to curb certain of
the issues experienced with GANs, such as the absence of adequate treatment for mixed data type variables, the difficult replication
of the correct single and multi-mode Gaussian distributions as well as long-tail distributions, the handling of imbalanced
or skewed variables, the absence of consensus on the optimal DP framework and the instability in training from weight
clipping.

It incorporates the following elements:

* a conditional generator and training-by-sampling introduced by CTGAN
* Wassertein loss, gradient penalty and the 5 to 1 discriminator to generator updates from WGAN
* information loss, which penalizes the discrepancy between statistics of the generated and real data (specifically, the mean and standard deviation)
* downstream loss, through an auxiliary classifier or regressor in parallel to the discriminator. The downstream loss penalizes the loss in ML utility measured by the difference in the synthesized and predicted values in the downstream ML task.
* mode-specific normalization in the preprocessing
* the use of Renyi differential privacy


Clover implementation
---------------------

The implementation in Clover is based on the code provided by the authors of CTAB-GAN-Plus. The DP version, however,
is not directly from the authors' repository but rather Opacus was used to integrate differential privacy for both
the discriminator and the downstream classifier, while the generator's privacy is achieved through the discriminator.

.. code-block::

    class CTABGANGenerator(Generator):
        """
        Wrapper for the GAN-based synthesizer presented in the paper CTAB-GAN+: Enhancing Tabular Data Synthesis, by
        Zhao & al.
        https://github.com/Team-TUD/CTAB-GAN-Plus

        See article for more information:
        https://arxiv.org/abs/2204.00401

        :param df: the data to synthesize
        :param metadata: a dictionary containing the list of **continuous** and **categorical** variables.
            These should comprehend all the columns to synthesize, including the columns in "mixed", "log", and
            "integer".
        :param random_state: for reproducibility purposes
        :param generator_filepath: the path of the generator to sample from if it exists
        :param mixed_columns: dictionary of "mixed" column names with corresponding categorical modes. Mixed columns are
            mostly continuous columns while one value or more - modes - hold another meaning (ex: 0).
        :param log_columns: list of skewed exponential numerical columns. These columns will go through a log transform.
        :param integer_columns: list of numeric columns without floating numbers. These columns will be rounded in the
            sampling step.
        :param class_dim: size of each desired linear layer for the auxiliary classifier
        :param random_dim: dimension of the noise vector fed to the generator
        :param num_channels: number of channels in the convolutional layers of both the generator and the discriminator
        :param l2scale: rate of weight decay used in the optimizer of the generator, discriminator and auxiliary classifier
        :param batch_size: batch size for training
        :param epochs: number of training epochs
        :param epsilon: the privacy budget of the differential privacy.
            One can specify how the budget is split between pre-processing and fitting the model.
            For example, epsilon = {"preprocessing": 0.1, "fitting": 0.9}.
            If a single float number is provided, half the budget is allocated for pre-processing
            and half for fitting.
        :param delta: target delta to be achieved for fitting (for differentially private model)
        :param max_grad_norm: the maximum norm of the per-sample gradients.
            Any gradient with norm higher than this will be clipped to this value. (for differentially private model)
        :param preprocess_metadata: specify the range (minimum and maximum) and optionally num_bins and decimals
            (to generate differentially private continuous samples) for all numerical columns
            and the distinct categories for categorical columns. This ensures that no further privacy leakage
            is happening (for differentially private model).
            For example, preprocess_metadata = {"col1": {"min": 0, "max": 1}, "col2": {"categories": ["cat1", "cat2"]}}.
        """

References
-----------
.. container:: references csl-bib-body hanging-indent
  :name: refs


  .. container:: csl-entry
     :name: ref-zhao_2022


     Zilong Zhao, Aditya Kunar, Robert Birke, & Lydia Y. Chen. 2022.
     "CTAB-GAN+: Enhancing Tabular Data Synthesis." arXiv. http://arxiv.org/abs/2204.00401.

