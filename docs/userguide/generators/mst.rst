Private-PGM - MST
==================

Introduction
------------

McKenna (2021) introduces a general approach for differentially private synthetic
data generation, which involves selecting low-dimensional marginals, adding
noise to measure them, and generating synthetic data that preserves these
marginals. This approach includes three high-level steps as follows. First, a
domain expert familiar with the data and its use cases can specify the set of
queries, or they can be automatically determined by an algorithm ("query
selection"). The selected queries are important because they will ultimately
determine the statistics for which the synthetic data preserves accuracy. After
the queries are set, the privacy is augmented with a noise-addition mechanism
such as the Gaussian mechanism, and noisy measurements are obtained ("query
measurement"). Finally, these measurements are processed to estimate a
high-dimensional data distribution and generate synthetic data ("synthetic data
generation").

The author's application of the framework is called MST, as it relies on a
Maximum Spanning Tree to perform the first step of selecting marginals. The last
step of synthetic data generation is performed by the algorithm *Private-PGM*, which
is also the name of the library developed by the authors, which provides tools for
the end-to-end generation of data following the method just described.

The main idea behind *Private-PGM* is to construct a Probabilistic Graphical
Model (PGM) that captures the dependencies between attributes in the data.
This model is then used to generate synthetic data that maintains the
correlations between attributes while satisfying differential privacy
guarantees. This approach is particularly effective in preserving the
statistical properties of the original data and has been successfully applied
in various domains, including health record data, where it outperformed existing
models in terms of data quality and model performance (Torfi, 2022).
To use Private-PGM for synthetic data generation, you can refer to the
`GitHub repository <https://github.com/ryan112358/private-pgm>`_ which
provides an implementation of the tools described in the paper.

Below is the pseudo code of the MST algorithm.

.. code-block:: python

   # Pseudo code for Private-PGM algorithm with MST

   # Step 1: Selection of marginals
   function SelectMarginals(data, 1way_marginals_log, privacy_budget):
   # This function returns the final set of pairs of attributes (i, j) to measure. It's a differentially private version of Kruskal's algorithm for computing a maximum spanning tree.
       estimated_2way_marginals = estimate_marginals(data, 1way_marginals_log)
       L1_errors = compute_L1_errors(data, estimated_2way_marginals)
       G_attributes = create_graph(data, L1_errors)
       pairs_to_measure = []
       for pair of attributes (i, j) in G:
           if i and j are in different connected components of G:
               # The exponential mechanism selects a highly weighted edge while respecting DP
               weight = exponential_mechanism(L1_errors)
               pairs_to_measure.append((i,j))
       return pairs_to_measure

   # Step 2: Measurement of marginals
   function MeasureMarginals(data, pairs_to_measure):
       noisy_marginals = []

       for pair in pairs_to_measure:
           noisy_marginal = compute_marginal(data, pair) + noise
           noisy_marginals.append(noisy_marginal)

       return noisy_marginals

   # Step 3: Generation of synthetic data
   function GenerateSyntheticData(noisy_marginals):
       # Infer a data distribution that best explains the marginals with PGM
       P = estimate_distribution_PGM(noisy_marginals)
       # Generate synthetic data from the PGM
       synthetic_data = sampleFromPGM(P)
       return synthetic_data


The Maximum Spanning Tree algorithm is used in the selection of marginals. The
classic version of Kruskal's algorithm for obtaining a Maximum Spanning Tree is
adapted by selecting a highly weighted edge through the exponential mechanism
rather than choosing the highest weighted edge.

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
