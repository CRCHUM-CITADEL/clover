Utility Metrics
===============

High-quality synthetic data should accurately capture the properties of the real data and be able to replicate the analysis performed on the real data. In Clover, a variety of general and specific utility metrics were implemented. General metrics provide an overall evaluation of synthetic data quality, while specific metrics are often used to assess data quality for specific tasks. The utility metrics available in Clover are summarized in Table 1.

.. list-table:: Table 1. Utility metrics in Clover
   :widths: 20 80
   :header-rows: 1

   * - Category
     - Metric
   * - Univariate
     - Continuous & categorical consistency
   * -
     - Continuous & categorical statistics
   * -
     - Hellinger distance
   * -
     - Kullback-Leibler divergence
   * - Bivariate
     - Pairwise Pearson correlation difference
   * -
     - Pairwise Chi-square correlation difference
   * - Population
     - Distinguishability
   * -
     - Cross learning (regression & classification)
   * - Application
     - Prediction (regression & classification)
   * -
     - F-Score for binary classification with continuous variables
   * -
     - Feature importance

Univariate Metrics
------------------

Univariate metrics are used to assess the ability of each individual variable in the synthetic data to capture the structure and fundamental aggregated statistics of the corresponding variable in real data. The univariate metrics implemented in Clover are as follows:

- Continuous & categorical consistency: This metric evaluates whether the values of continuous and categorical variables in the synthetic data adhere to the bounds and categories observed in the real data.
- Continuous & categorical statistics: For continuous variables, these statistics assess the congruence of median and interquartile range (IQR) with those of the real data. Categorical statistics examine whether the synthetic data align with the support and frequency coverage of categorical variables observed in the real data.
- Hellinger distance: This quantifies the similarity between the distributions of the real and synthetic data.
- Kullback-Leibler divergence: This measures the relative entropy in information represented by the distributions of the real and synthetic data.

Bivariate Metrics
-----------------

Bivariate metrics are used to evaluate whether synthetic data maintains the same correlations between variables as the real data. In Clover, the following bivariate metrics have been implemented:

- Pairwise Pearson correlation difference: This metric assesses the preservation of pairwise Pearson correlations between continuous variables.
- Pairwise Chi-square correlation difference: It evaluates the preservation of pairwise relationships among categorical variables.

Population Metrics
------------------

Population metrics assess the similarity of the entire distribution between synthetic and real data. In Clover, the following population metrics have been implemented:

- Distinguishability: This metric evaluates the similarity of real and synthetic data by training a model to differentiate between them.
- Cross learning (regression & classification): It assesses the ability of synthetic data to capture the statistical dependency structure of the real data. This is done by building classification or regression models to predict each categorical or continuous variable using all the remaining variables.

Application Metrics
-------------------

Application metrics are commonly used to evaluate the performance of task-specific applications. These metrics verify whether synthetic data can replicate results obtained from real data in specific analyses. The following application metrics have been implemented in Clover:

- Prediction (regression & classification): This assesses whether synthetic data exhibits behavior comparable to real data in regression or classification tasks.
- F-Score for binary classification with continuous variables: This feature selection technique gauges the discriminatory capacity of a feature. It evaluates the similarities of F-scores for each feature in the real and synthetic datasets for binary classification problems with continuous variables.
- Feature importance: This metric quantifies the reduction in model performance when a feature value is randomly shuffled. It assesses whether the predictive significance of each feature remains consistent across the task.

References
----------

- `Goncalves, A., Ray, P., Soper, B., Stevens, J., Coyle, L., & Sales, A. P. (2020). Generation and evaluation of synthetic patient data. BMC medical research methodology, 20, 1-40. <https://link.springer.com/article/10.1186/s12874-020-00977-1>`_

- `Dankar, F. K., Ibrahim, M. K., & Ismail, L. (2022). A multi-dimensional evaluation of synthetic data generators. IEEE Access, 10, 11147-11158. <https://ieeexplore.ieee.org/abstract/document/9686689>`_

- `El Emam, K., Mosquera, L., Fang, X., & El-Hussuna, A. (2022). Utility metrics for evaluating synthetic health data generation methods: validation study. JMIR medical informatics, 10(4), e35734. <https://medinform.jmir.org/2022/4/e35734>`_

- `Chen, Y. W., & Lin, C. J. (2006). Combining SVMs with various feature selection strategies. Feature extraction: foundations and applications, 315-324. <https://link.springer.com/chapter/10.1007/978-3-540-35488-8_13>`_

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/utility_report.ipynb
