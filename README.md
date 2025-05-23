# Clover: Synthetic Health Data Generation and Validation Library

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Tests with Pytest](https://img.shields.io/badge/Tests-Pytest-green)](https://pytest.org)
[![codecov](https://codecov.io/gh/CRCHUM-CITADEL/clover/branch/main/graph/badge.svg)](https://codecov.io/gh/CRCHUM-CITADEL/clover)
[!Pytest Status](https://github.com/CRCHUM-CITADEL/clover/actions/workflows/pytest.yml/badge.svg)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io)
[![Docs: Sphinx](https://img.shields.io/badge/Docs-Sphinx-blue)](https://www.sphinx-doc.org)

</div>

Advances in health research are constrained by the availability of data. Indeed, access to a large amount of data from
different sources is a key factor to increase the generalizability in health research and thus improve healthcare for the population.

Public and pre-processed data do not reflect the real-world.
Synthetic data in healthcare refers to artificially generated datasets that mimic the statistical properties and relationships found in real-world patient data without containing any personally identifiable information. This data is created using advanced algorithms and machine learning techniques, making it a valuable resource for healthcare professionals and researchers who require access to data-driven insights while adhering to stringent patient privacy regulations. Synthetic data can take various forms, ranging from fully synthetic datasets that contain no real records to partially synthetic datasets where only specific sensitive variables are replaced, and hybrid approaches that combine elements of both real and synthetic data. The methods used to generate synthetic data are also diverse, encompassing rule-based systems to sophisticated models.
While synthetic data is designed to mitigate the risks associated with using real patient data, we need to recognize that it does not entirely eliminate privacy risks if not done properly. Several potential vulnerabilities and limitations can lead to the disclosure of sensitive information, even in artificially generated datasets.
It is essential to establish best practices to mitigate the risks of privacy breach and information loss.

Introducing Clover, a comprehensive library designed for the critical assessment of synthetic data generation. Clover evaluates the quality of both the generated synthetic data and the methods used to create it based on the degree to which information from the original data is preserved and the level of privacy protection afforded. Recognizing the inherent trade-off between these aspects, Clover aims to facilitate the creation of synthetic data that effectively balances the utility of real-world data with the imperative of safeguarding patient privacy.

## Table of Contents

* [Useful Links](#useful-links)
* [Current Features](#current-features)
* [Usage](#usage)
  - [Requirements](#requirements)
  - [Installation](#installation)
* [Quickstart](#quickstart)
* [Join Our Community](#join-our-community)
* [Ongoing Work - Next Steps](#ongoing-work---next-steps)

## Useful Links

* [Documentation](#documentation)
* [Github Repository](https://github.com/CRCHUM-CITADEL/clover)

## Documentation

We shall keep the repository private during the first stage of development.
The documentation is therefore not yet available as a web page. To browse it, please follow the following steps:

* `git clone git@github.com:CRCHUM-CITADEL/clover.git` or `git clone https://github.com/CRCHUM-CITADEL/clover`
* `git switch gh-pages` (a new local branch will automatically be created from the remote `gh-pages` branch)
* open the file `index.html` in your browser to visualize the documentation home page

Nb: The branch `gh-pages` is recreated each time the main branch is modified on Github.


## Current Features

* Synthetic data generators incorporating integrated differential privacy, supporting continuous and categorical variables (unique identifiers are not handled):
   - [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer)
   - [Synthpop](https://github.com/hazy/synthpop)
   - [SMOTE](https://imbalanced-learn.org/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn)
   - [MST (Maximum Spanning Tree)](https://github.com/ryan112358/private-pgm/tree/master)
   - [CTGAN](https://github.com/sdv-dev)
   - [TVAE](https://github.com/sdv-dev)
   - [CTAB-GAN+](https://github.com/Team-TUD/CTAB-GAN-Plus)
   - [FinDiff](https://github.com/sattarov/FinDiff)
* Utility and privacy reports to assess the fidelity of the synthetic data:
   - Summary table
   - Detailed report with figures
* The following utility metrics are implemented:
   - Univariate metrics 
     - Continuous & categorical consistency 
     - Continuous & categorical statistics 
     - Hellinger distance 
     - Kullback-Leibler divergence
   - Bivariate metrics 
     - Pairwise Pearson Correlation Difference 
     - Pairwise Chi-square correlation difference
   - Population metrics 
     - Distinguishability 
     - Cross learning (regression & classification)
   - Application metrics 
     - Prediction (regression & classification)
     - F-Score for binary classification with continuous variables only
     - Feature importance
* The following privacy metrics are implemented:
   - Reidentification metrics: Assess the risk of linking records in the synthetic data back to specific individuals in the original real dataset. 
     - Distance to Closest Record: Measures how similar each synthetic record is to its nearest neighbor in the real data, indicating potential for identifying near-duplicates.  
     - Nearest Neighbor Distance Ratio: Compares the distance to the nearest neighbor within the synthetic data to the distance to the nearest neighbor in the real data for synthetic points, highlighting if synthetic points are too close to real ones.
   - Membership inference attack (MIA): Evaluates how well an adversary can determine if a particular record was part of the original training dataset used to generate the synthetic data. 
     - GAN-Leaks: Specifically assesses the leakage of information from the training data in synthetic data generated by Generative Adversarial Networks (GANs).
     - Monte Carlo membership inference attack: A specific type of membership inference attack that uses Monte Carlo simulation to estimate the probability of a record being in the training data. 
     - Logan: Assesses the risk of membership inference by training a model to distinguish between the first and second generations of synthetic data. 
     - TableGan: Evaluates the vulnerability to membership inference by training both a discriminator (to distinguish between real and synthetic data) and a classifier (likely to predict whether a record was part of the training set).
     - Detector: Measures the susceptibility to membership inference by training a model to classify between the first generation of synthetic data and real data that was not used to generate the synthetic data.
     - Collision: Measures the frequency of identical or very similar records appearing in the synthetic dataset, which could indicate a privacy risk if unique real records are being replicated.
* Metareport to compare several synthetic datasets with respect to the metrics

See the [documentation](#documentation) for more details.

## Usage

### Requirements
All the required packages are available in the [requirements file](requirements.txt).
Clover has been tested on a Linux system running Python 3.8.10 and Python 3.10.

### Installation
The package is not yet available on pypi. You can clone the Github repository.
The branch 'main' contains the latest development version.

## Quickstart
To get started, we created 4 notebooks to guide you through the generation of synthetic data,
their associated utility and privacy reports and the hyperparameters tuning:
* [Synthetic data generation](notebooks/synthetic_data_generation.ipynb)
* [Utility report](notebooks/utility_report.ipynb)
* [Privacy report](notebooks/privacy_report.ipynb)
* [Tune hyperparameters](notebooks/tune_hyperparameters.ipynb)

To get the average summary metrics results for both utility and privacy at once, see the 
[combined report](notebooks/combined_report.ipynb) notebook. To compare several synthetic datasets 
with respect to a list of metrics, see the [metareport](notebooks/metareport.ipynb) notebook.

The notebooks are based on the 
[Breast Wisconsin Cancer WBCD dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29).

## Join Our Community
If you have any question, feature request or if you have encountered an issue, please open an issue on Github.

We also welcome any contribution to the project. 
The required packages for development can be found in the [dev-requirements file](dev-requirements.txt).
The documentation was generated with Sphinx.

## Ongoing Work - Next Steps
* Improve data coverage (direct identifiers, missing data, etc.)
* Add support for imaging data
* Improve the utility metrics (better discretisation, learning algorithms, etc.)
* Create a benchmark of the synthetic data generator in different settings
