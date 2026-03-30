TVAE
=====

Introduction
------------

Autoencoders are a type of artificial neural network used for learning efficient representations of data. They consist of an encoder network that maps the input data into a latent space representation and a decoder network that reconstructs the input data from the latent space. Variational autoencoders (VAEs) are a specific type of autoencoder trained to generate new data samples from the learned latent space in a way that takes into consideration the variability in the original sample and ensures continuity and smoothness of the generated samples. VAEs can be used for synthetic data generation by sampling from the learned latent space to produce new, artificial data that resembles the original data.

Latent space representation is one of the central aspects of VAEs that enables it to capture the underlying structure and variations present in the input data, condensing it into a continuous, low-dimensional space. This latent space has the essential characteristics of the dataset, facilitating the generation of new data points similar to those observed in the original dataset. By sampling from different regions within the latent space and decoding these samples, VAEs can generate synthetic data points that correspond to the statistical properties of the input data. In practical applications, VAEs find utility in various domains. In healthcare, for instance, VAEs can learn the statistical distributions of patient data and produce synthetic patient records that preserve key characteristics while safeguarding privacy. Furthermore, VAEs contribute to data augmentation efforts by generating synthetic data points to supplement existing datasets, thereby enhancing the performance of machine learning models.

The implementation that is used here is based on the `TVAE <https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py>`_ model for synthetic data generation. TVAE consists of an encoder and a decoder, where the encoder compresses the input data into a lower-dimensional latent space, and the decoder reconstructs the original data from this latent representation. The ``Encoder`` class takes as input the dimensions of the data, the size of each hidden layer (\ ``compress_dims``\ ), and the size of the output vector (\ ``embedding_dim``\ ). The encoder applies a sequence of linear transformations followed by ReLU activation functions to map the input data to the latent space. It then outputs the mean (\ ``mu``\ ), standard deviation (\ ``std``\ ), and logarithm of the variance (\ ``logvar``\ ) of the latent representation. The ``Decoder`` class takes as input the size of the input vector (\ ``embedding_dim``\ ), the size of each hidden layer (\ ``decompress_dims``\ ), and the dimensions of the original data. Similar to the encoder, the decoder applies a sequence of linear transformations followed by ReLU activation functions to reconstruct the original data from the latent space.
The loss function calculates the reconstruction loss and the Kullback-Leibler divergence, which measures the difference between the learned latent distribution and a standard normal distribution.

Below is the pseudo code for the `TVAE algorithm <https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py>`_\ :

.. code-block:: python

   1. Instantiate TVAE with hyperparameters
   2. Fit TVAE model to training data
   3. Generate synthetic data using TVAE model

   Initialize TVAE algorithm with hyperparameters:
   - embedding_dim
   - compress_dims
   - decompress_dims

   Define Encoder class:
   - Initialize with data_dim, compress_dims, embedding_dim
   - Produces mean, standard deviation, and logvariance 
   - Define forward method to encode input data

   Define Decoder class:
   - Initialize with embedding_dim, decompress_dims, data_dim
   - Define forward method to decode input data

   Define loss_function to compute loss:
   - Compute the reconstruction/divergence loss

   Define TVAE class:
   - Initialize with hyperparameters
   - Define fit method to train the TVAE model:
     - Transform input data
     - Initialize Encoder and Decoder
     - Iterate through epochs:
       - Iterate through batches:
         - Compute encoder output
         - Compute decoder output
         - Compute loss
         - Update model parameters

   Define sample method to generate synthetic data:
     - Generate synthetic data based on input samples

Here's a breakdown of each step:


#. 
   **Encoder Class (Initialization and Forward Pass)**\ :


   * Initialize the Encoder class with ``data_dim``\ , ``compress_dims``\ , and ``embedding_dim``.
   * A sequence of linear layers followed by ReLU activation functions based on the ``compress_dims``.
   * Define linear layers to produce the mean ``mu`` and log variance ``logvar`` of the latent space.
   * Implement the ``forward`` method to encode the input data, where the input passes through the sequential layers, and then produces ``mu``\ , ``std`` (standard deviation), and ``logvar``.

#. 
   **Decoder Class (Initialization and Forward Pass)**\ :


   * Initialize the Decoder class with ``embedding_dim``\ , ``decompress_dims``\ , and ``data_dim``.
   * Construct a sequence of linear layers followed by ReLU activation functions based on the ``decompress_dims``.
   * Add a linear layer at the end to produce the output data.
   * Additionally, create a ``sigma`` parameter to model the uncertainty of the reconstructed data.
   * Implement the ``forward`` method to decode the input vector, where the input passes through the sequential layers to produce the reconstructed data.

#. 
   **Loss Function**\ :


   * Define a loss function to compute the reconstruction loss/Kullback-Leibler divergence loss.
   * Iterate over the output_info to compute the loss for each column of the data.
   * Compute the reconstruction loss and Kullback-Leibler divergence loss, and return their sum.

#. 
   **TVAE Class (Initialization, Fit, Sample, and Set Device)**\ :


   * Initialize the TVAE class with various hyperparameters including ``embedding_dim``\ , ``compress_dims``\ , ``decompress_dims``\ , etc.
   * Define the ``fit`` method to train the TVAE model on the input data:

     * Iterate through epochs and batches, compute encoder and decoder outputs, compute loss, and update model parameters accordingly.

   * Define the ``sample`` method to generate synthetic data similar to the training data using the trained Decoder:

     * Generate noise vectors, pass them through the Decoder, and transform the generated data back to the original space.

Clover implementation
---------------------

.. code-block::

   """
   Also a wrapper of a data synthesizer available in the SDV package.
   The synthesizer is TVAE, a VAE for tabular data.
   https://github.com/sdv-dev

   :param df: the data to synthesize
   :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
   :param random_state: for reproducibility purposes
   :param generator_filepath: the path of the generator to sample from if it exists
   :param discriminator_steps: the number of discriminator updates to do for each generator update.
   :param epochs: the number of training epochs.
   :param batch_size: the batch size for training.
   :param compress_dims: the size of the hidden layers in the encoder.
   :param decompress_dims: the size of the hidden layers in the decoder.
   """
   Steps include:
   Preparing the parameters to train the generator.
   Train the generator and save it.
   Generate samples using the synthesizer trained on the real data.



References
----------


* https://github.com/sdv-dev/SDV
* https://github.com/sdv-dev/SDV/blob/main/sdv/single_table/ctgan.py
* https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py
* `TVAE Model <https://sdv.dev/SDV/user_guides/single_table/tvae.html>`_
* `TVAESynthesizer <https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer>`_
* `Modeling Tabular data using Conditional GAN <https://arxiv.org/pdf/1907.00503.pdf>`_
* `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_
* `Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition <https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/>`_ 
