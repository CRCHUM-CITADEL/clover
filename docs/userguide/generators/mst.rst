Maximum Spanning Tree (MST)
===========================

Introduction
------------

McKenna (2021) introduces a general approach for differentially private synthetic data generation, which involves selecting low-dimensional marginals, adding noise to measure them, and generating synthetic data that preserves these marginals. This approach includes three high-level steps as follows. First, a domain expert familiar with the data and its use cases can specify the set of queries, or they can be automatically determined by an algorithm ("query selection"). The selected queries are important because they will ultimately determine the statistics for which the synthetic data preserves accuracy. After the queries are set, the privacy is augmented with a noise-addition mechanism such as the Gaussian mechanism, and noisy measurements are obtained ("query measurement"). Finally, these measurements are processed to estimate a high-dimensional data distribution and generate synthetic data ("synthetic data generation").
Here, this procedure is based on the Maximum Spanning Tree (MST) functionality from *Private-PGM* Module.

The main idea behind *Private-PGM* is to construct a Probabilistic Graphical Model (PGM) that captures the dependencies between attributes in the data. This model is then used to generate synthetic data that maintains the correlations between attributes while satisfying differential privacy guarantees.
This approach is particularly effective in preserving the statistical properties of the original data and has been successfully applied in various domains, including health record data, where it outperformed existing models in terms of data quality and model performance (Torfi, 2022).
To use Private-PGM for synthetic data generation, you can refer to the `GitHub repository <https://github.com/ryan112358/private-pgm>`_ which provides an implementation of the tools described in the paper.

Below is the pseudo code for the Private-PGM algorithm can be outlined as follows:

.. code-block:: python

   # Pseudo code for Private-PGM algorithm with MST

   # Step 1: Measurement
   function MeasureMarginals(data, marginals, epsilon):
       noisy_marginals = {}
       for marginal in marginals:
           true_value = computeMarginal(data, marginal)
           noise = generateNoise(epsilon)  # Laplace or Gaussian noise
           noisy_marginals[marginal] = true_value + noise
       return noisy_marginals

   # Step 2: Synthetic Data Generation
   function GenerateSyntheticData(noisy_marginals):
       # Construct a PGM based on the noisy marginals
       pgm = constructPGMwithMST(noisy_marginals)
       # Generate synthetic data from the PGM
       synthetic_data = sampleFromPGM(pgm)
       return synthetic_data

      # the constructPGMwithMST function using MST
      function constructPGMwithMST(noisy_marginals):
          # Use Maximum Spanning Tree algorithm to determine the edges of the PGM
          mst_edges = maximumSpanningTreeAlgorithm(noisy_marginals)
          # Construct the PGM based on the noisy marginals and the MST edges
          return pgm

   # Main function
   function PrivatePGM(data, marginals, epsilon):
       noisy_marginals = MeasureMarginals(data, marginals, epsilon)
       synthetic_data = GenerateSyntheticData(noisy_marginals)
       return synthetic_data

Here's a breakdown of each step:

Let's break down the pseudo code for the Private-PGM algorithm with the use of the Maximum Spanning Tree (MST) algorithm:


#. 
   **Measurement (MeasureMarginals function):**


   * **Input:** Original data (\ ``data``\ ), a set of marginals to be preserved (\ ``marginals``\ ), and privacy parameter (\ ``epsilon``\ ).
   * **Output:** Noisy marginals for the specified variables.
   * **Procedure:**

     * For each specified marginal variable in the input set ``marginals``\ :

       * Compute the true marginal value (\ ``true_value``\ ) based on the original data using the ``computeMarginal`` function.
       * Generate Laplace or Gaussian noise (\ ``noise``\ ) using the ``generateNoise`` function with privacy parameter ``epsilon``.
       * Add the noise to the true value to obtain the noisy marginal.
       * Store the noisy marginal in the ``noisy_marginals`` dictionary.

     * Return the dictionary of noisy marginals.

#. 
   **Synthetic Data Generation (GenerateSyntheticData function):**


   * **Input:** Noisy marginals obtained from the measurement step (\ ``noisy_marginals``\ ).
   * **Output:** Synthetic data generated based on the constructed Probabilistic Graphical Model (PGM).
   * **Procedure:**

     * The ``constructPGMwithMST`` function involves using the Maximum Spanning Tree algorithm to create the graphical structure of the model.
     * Generate synthetic data (\ ``synthetic_data``\ ) by sampling from the constructed PGM using the ``sampleFromPGM`` function.

   * Return the synthetic data.

#. 
   **Maximum Spanning Tree (constructPGMwithMST function):**


   * The ``constructPGMwithMST`` function is responsible for constructing the Probabilistic Graphical Model (PGM) based on the noisy marginals using the Maximum Spanning Tree algorithm.
   * The Maximum Spanning Tree algorithm helps determine the edges of the PGM, representing significant relationships among variables.
   * The resulting PGM structure is used to guide the generation of synthetic data.

#. 
   **Main Function (PrivatePGM function):**


   * **Input:** Original data (\ ``data``\ ), a set of marginals to be preserved (\ ``marginals``\ ), and privacy parameter (\ ``epsilon``\ ).
   * **Output:** Synthetic data generated while preserving marginals.
   * **Procedure:**

     * Call the ``MeasureMarginals`` function to obtain the noisy marginals.
     * Call the ``GenerateSyntheticData`` function with the obtained noisy marginals to generate synthetic data.
     * Return the synthetic data.

The Maximum Spanning Tree algorithm is specifically utilized in the ``constructPGMwithMST`` function to determine the graphical structure of the Probabilistic Graphical Model (PGM) based on the noisy marginals. This structure is then used to generate synthetic data while preserving statistical properties specified by the marginals.

Clover implementation
---------------------

.. code-block::

    MSTGenerator(Generator):
   """
   Wrapper of the Maximum Spanning Tree (MST) method from Private-PGM repo:
   https://github.com/ryan112358/private-pgm/tree/master.

   :cvar name: the name of the metric
   :vartype name: str

   :param df: the data to synthesize
   :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
   :param random_state: for reproducibility purposes
   :param generator_filepath: the path of the generator to sample from if it exists
   :param epsilon: the privacy budget of the differential privacy
   :param delta: the failure probability of the differential privacy
   """


Steps include:

#.
   Preparing the parameters to train the generator.
#.
   Define and save the MST parameters. The fit is executed with the sampling.
#.
   Generate samples using the MST method.s







References:
-----------


* `Winning the NIST Contest: A scalable and general approach to differentially private synthetic data <https://arxiv.org/pdf/2108.04978.pdf>`_
* `Priv Syn:Differentially Private Data Synthesis <https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf>`_
* 
  `Differentially private synthetic medical data generation using convolutional GANs <https://www.sciencedirect.com/science/article/abs/pii/S0020025521012391>`_

* 
  https://github.com/ryan112358/private-pgm

* https://github.com/BorealisAI/private-data-generation
* https://github.com/alan-turing-institute/reprosyn
