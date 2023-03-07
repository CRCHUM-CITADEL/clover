# Clover: Synthetic Health Data Generation and Validation Library

Advances in health research are constrained by the availability of data. Indeed, access to a large amount of data from
different sources is a key factor to increase the generalizability of the machine learning algorithms and validate them
and thus improve healthcare for the population.

Public and pre-processed data do not reflect the real-world.
Synthetic data, which preserve the properties of the original dataset while overcoming privacy risks
since the information is no longer personal, hold promise.
However, the evidence regarding their utility and security remains unclear.
For widespread adoption of synthetic data, both by the general public and by potential users,
it is essential to establish best practices to mitigate the risks of privacy breach and information loss.

The goal of this project is therefore to provide means to perform a comprehensive study on synthetic data generation.
The quality of the synthetic data and their generator will be evaluated on two criteria:
the preservation of information and privacy. A trade-off between these two aspects is necessary in order to
preserve the properties of the real data without compromising the privacy of the patients.

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

* [Documentation]()
* [Github Repository](https://github.com/CRCHUM-CITADEL/clover)

## Current Features

* Synthetic data generators, supporting continuous and categorical variables (unique identifiers are not handled):
   - [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer)
   - [Synthpop](https://github.com/hazy/synthpop) with random or Particle Swarm Optimization search to tune the variables order
* Utility report to assess the fidelity of the synthetic data:
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

See the documentation for more details.

## Usage

### Requirements
All the required packages are available in the [requirements file](requirements.txt).
Clover has been tested on a Linux system running Python 3.8.10.

### Installation
The package is not yet available on pypi. You can clone the Github repository.
The branch 'main' contains the latest development version.

### Quickstart
To get started, we created two notebooks to guide you through the generation of synthetic data
and their associated utility report:
* [Synthetic data generation](notebooks/synthetic_data_generation.ipynb)
* [Utility report](notebooks/utility_report.ipynb)

The notebooks are based on the 
[Breast Wisconsin Cancer WBCD dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29).

### Join Our Community
If you have any question, feature request or if you have encountered an issue, please open an issue on Github.

We also welcome any contribution to the project. The required packages for development can be found in the [dev-requirements file](dev-requirements.txt)

### Ongoing Work - Next Steps
* Implement more generators
* Optimise the generators with hyperparameters search
* Integrate Differential Privacy in to each generator
* Implement privacy metrics
* Improve data coverage (direct identifiers, missing data, etc.)
* Improve the utility metrics (better discretisation, learning algorithms, etc.)
* Create a benchmark of the synthetic data generator in different settings
