Privacy Metrics
===============

While the generated synthetic data should preserve the properties of the real data, it is also crucial that the synthetic data does not reveal any information about the individuals in the real data. In Clover, several privacy metrics have been implemented to evaluate the privacy-preserving integrity of the synthetic data. These privacy metrics are summarized in Table 1.

.. list-table:: Table 1. Privacy metrics in Clover
   :widths: 40 60
   :header-rows: 1

   * - Category
     - Metric
   * - Reidentification
     - Distance to closest record (DCR)
   * -
     - Ratio match
   * -
     - Nearest neighbor distance ratio (NNDR)
   * - Membership inference attack (MIA)
     - GAN-Leaks
   * -
     - Monte Carlo membership inference attack
   * -
     - Logan
   * -
     - TableGan
   * -
     - Detector
   * -
     - Collision

Reidentification
----------------

Reidentification metrics assess the reidentification risk using distance-based algorithms. They provide an overview of the privacy implications of synthetic data. If the distance between real and synthetic data is too small, there may be a risk of revealing sensitive information from the training data. Conversely, if the distance is too large, the quality of the synthetic data might be poor. The reidentification metrics implemented in Clover include:

- Distance to closest record (DCR): This quantifies the proximity between synthetic data points and their closest counterparts in the real data.
- Ratio match: It calculates the proportion of records with a DCR below a predefined threshold.
- Nearest neighbor distance ratio (NNDR): This assesses the ratio between the Gower distance for the nearest and the second nearest neighbor in the real data for any corresponding synthetic record.

Membership inference attack (MIA)
---------------------------------

Membership disclosure occurs when an adversary gains access to data from the population that the synthetic data was generated from. Membership inference attack (MIA) is often achieved by training a machine learning model to infer whether a record in the population was used for training a synthetic data generator. In Clover, a range of state-of-the-art MIA models have been adapted and implemented:

- GAN-Leaks: This method infers membership based on the distance to closest record (DCR) for each record in the real data relative to its counterpart in the synthetic data.
- Monte Carlo membership inference attack: It infers membership based on the number of neighbors within the synthetic data for each record in the real data.
- Logan: Membership inference is achieved by training a model to classify the 1st and 2nd generation synthetic data.
- TableGan: Membership inference is achieved by training a discriminator and a classifier.
- Detector: Membership inference is achieved by training a model to classify the 1st generation synthetic data and real data that were not utilized in generating the synthetic data.
- Collision: This method trains a model to classify whether each record in the synthetic data collides with a record in the real data that was used to generate the synthetic data.

References
----------

- `Zhao, Z., Kunar, A., Birke, R., & Chen, L. Y. (2021, November). Ctab-gan: Effective table data synthesizing. In Asian Conference on Machine Learning (pp. 97-112). PMLR. <https://proceedings.mlr.press/v157/zhao21a>`_

- `Chen, D., Yu, N., Zhang, Y., & Fritz, M. (2020, October). Gan-leaks: A taxonomy of membership inference attacks against generative models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security (pp. 343-362). <https://dl.acm.org/doi/abs/10.1145/3372297.3417238>`_

- `Hilprecht, B., Härterich, M., & Bernau, D. (2019). Monte carlo and reconstruction membership inference attacks against generative models. Proceedings on Privacy Enhancing Technologies. <https://petsymposium.org/popets/2019/popets-2019-0067.php>`_

- `Hayes, J., Melis, L., Danezis, G., & De Cristofaro, E. (2017). Logan: Membership inference attacks against generative models. arXiv preprint arXiv:1705.07663. <https://arxiv.org/abs/1705.07663>`_

- `Park, N., Mohammadi, M., Gorde, K., Jajodia, S., Park, H., & Kim, Y. (2018). Data synthesis based on generative adversarial networks. arXiv preprint arXiv:1806.03384. <https://arxiv.org/abs/1806.03384>`_

- `Olagoke, L., Vadhan, S., & Neel, S. (2023). Black-Box Training Data Identification in GANs via Detector Networks. arXiv preprint arXiv:2310.12063. <https://arxiv.org/abs/2310.12063>`_

- `Hu, A., Xie, R., Lu, Z., Hu, A., & Xue, M. (2021, November). Tablegan-mca: Evaluating membership collisions of gan-synthesized tabular data releasing. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (pp. 2096-2112). <https://dl.acm.org/doi/abs/10.1145/3460120.3485251>`_

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/privacy_report.ipynb
