**********************************
Welcome to Clover's documentation!
**********************************

-------------------------------------------------------
Synthetic Health Data Generation and Validation Library
-------------------------------------------------------

Advances in health research are constrained by the availability of data. Indeed, access to a large amount of data from different sources is a key factor to increase the generalizability of the machine learning algorithms and validate them and thus improve healthcare for the population.

Public and pre-processed data do not reflect the real-world. Synthetic data, which preserve the properties of the original dataset while overcoming privacy risks since the information is no longer personal, hold promise. However, the evidence regarding their utility and security remains unclear. For widespread adoption of synthetic data, both by the general public and by potential users, it is essential to establish best practices to mitigate the risks of privacy breach and information loss.

The goal of this project is therefore to provide means to perform a comprehensive study on synthetic data generation. The quality of the synthetic data and their generator will be evaluated on two criteria: the preservation of information and privacy. A trade-off between these two aspects is necessary in order to preserve the properties of the real data without compromising the privacy of the patients.

Useful Links
============

* `Github Repository <https://github.com/CRCHUM-CITADEL/clover>`_

Current Features
================

* Synthetic data generators, supporting continuous and categorical variables (unique identifiers are not handled):
   - `DataSynthesizer <https://github.com/DataResponsibly/DataSynthesizer>`_
   - `Synthpop <https://github.com/hazy/synthpop>`_
   - `SMOTE (Synthetic Minority Oversampling TEchnique) <https://imbalanced-learn.org/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn>`_
   - `MST (Maximum Spanning Tree) <https://github.com/ryan112358/private-pgm/tree/master>`_
   - `CTGAN <https://github.com/sdv-dev>`_
   - `TVAE <https://github.com/sdv-dev>`_
   - `CTAB-GAN+ <https://github.com/Team-TUD/CTAB-GAN-Plus>`_
   - `TabDDPM <https://github.com/hazy/synthpop>`_
* Utility report to assess the fidelity of the synthetic data:
   - Summary table
   - Detailed report with figures
* The following utility metrics are implemented:
   - Univariate metrics
      + Continuous & categorical consistency
      + Continuous & categorical statistics
      + Hellinger distance
      + Kullback-Leibler divergence
   - Bivariate metrics
      + Pairwise Pearson Correlation Difference
      + Pairwise Chi-square correlation difference
   - Population metrics
      + Distinguishability
      + Cross learning (regression & classification)
   - Application metrics
      + Prediction (regression & classification)
      + F-Score for binary classification with continuous variables only
      + Feature importance
* The following privacy metrics are implemented:
   - Reidentification metrics
      + Distance to Closest Record
      + Nearest Neighbor Distance Ratio
* Metareport to compare several synthetic datasets with respect to the metrics

See the documentation of each component in the User Guide section for more details.

Ongoing Work - Next Steps
=========================

* Integrate Differential Privacy in to each generator
* Implement privacy metrics
* Improve data coverage (direct identifiers, missing data, etc.)
* Improve the utility metrics (better discretization, learning algorithms, etc.)
* Create a benchmark of the synthetic data generator in different settings


.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   home/clover
   home/installation
   home/quickstart
   home/getting_involved
   home/license


.. toctree::
   :caption: User Guide
   :maxdepth: 2
   :hidden:

   userguide/introduction
   userguide/generators/modules
   userguide/metrics/modules
   userguide/optimization/modules

.. toctree::
   :caption: API
   :maxdepth: 2
   :hidden:

   API/generators/modules
   API/metrics/modules
   API/optimization/modules
   API/utils/modules
   API/tests/modules

.. toctree::
   :caption: Tutorial
   :maxdepth: 2
   :hidden:

   tutorial/synthetic_data_generation.nblink
   tutorial/utility_report.nblink
   tutorial/privacy_report.nblink
   tutorial/tune_hyperparameters.nblink
   tutorial/combined_report.nblink
   tutorial/metareport.nblink
