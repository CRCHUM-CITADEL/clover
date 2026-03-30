SMOTE
=====

Introduction
------------

SMOTE (Synthetic Minority Over-sampling Technique) is a data augmentation distance-based technique. It addresses the problem of class imbalance in machine learning by oversampling the minority class.

Primarily SMOTE has been used in medical applications to improve classification performance over imbalanced medical datasets. However, it found applications beyond its initial domain. Several studies used SMOTE for synthetic data generation in medical applications. For instance, one study applied SMOTE to generate synthetic data for predictive models in low-middle-income countries, while another utilized SMOTE to generate high-fidelity synthetic patient data for assessing machine learning healthcare software. Additionally, a novel algorithm called SMOTE-ENC was proposed to generate synthetic data for nominal and continuous features in medical imbalanced data. Building on the SMOTE capabilities, another approach called Data Collaboration (DC) analysis has emerged, enabling privacy-conscious joint analysis across diverse institutions. This method aggregates dimensionality-reduced representations and facilitates comprehensive analysis through collaborative representations, all while safeguarding the original data.


Algorithm
---------

SMOTE draws new observations from the line formed between two samples closed together in the feature space. Specifically, the algorithm:

- randomly selects a sample from the minority class,
- randomly chooses one of its k nearest neighbors (typically k=5),
- draws a new synthetic observation at a random point between the two selected samples in the feature space.

There are some variations of SMOTE, such as ADASYN (Adaptive Synthetic Sampling Method), which is a modification of SMOTE that generates more synthetic examples near the boundary of the minority class. In the literature SMOTE is presented as a powerful solution for imbalanced data, but it has a drawback. It does not consider the majority class while creating synthetic examples, which can cause issues where there is a strong overlap between the classes. Therefore, the original SMOTE paper suggests combining oversampling (SMOTE) with undersampling of the majority class. To better understand the algorithm, below is a pseudocode:


.. code-block:: none

    function SMOTE(dataset, minority_class, N, k):
    synthetic_samples = []

    for each sample in minority_class:
        neighbors = k_nearest_neighbors(sample, dataset, k)

        for i in range(N):
            neighbor = randomly_select_neighbor(neighbors)
            synthetic_sample = generate_synthetic_sample(sample, neighbor)
            synthetic_samples.append(synthetic_sample)

    return synthetic_samples

    function k_nearest_neighbors(sample, dataset, k):
        distances = compute_distances(sample, dataset)
        sorted_neighbors = sort_by_distance(distances)
        return sorted_neighbors[:k]

    function randomly_select_neighbor(neighbors):
        return randomly_pick_one_neighbor(neighbors)

    function generate_synthetic_sample(sample, neighbor):
        synthetic_sample = {}

        for each feature in sample:
            difference = neighbor[feature] - sample[feature]
            synthetic_sample[feature] = sample[feature] + random_uniform(0, 1) * difference

        return synthetic_sample

In this pseudocode:

- `dataset` is the entire dataset.
- `minority_class` is the class that is in the minority (you apply SMOTE to balance it with the majority class).
- `N` is the number of synthetic samples to generate for each original minority class sample.
- `k` is the number of nearest neighbors to consider when generating synthetic samples.

As mentioned above basic steps of SMOTE involve selecting a sample from the minority class, finding its k-nearest neighbors and creating synthetic samples by combining features from the selected sample and its neighbors.


Clover implementation
---------------------

Implementation of the SMOTE generator in the Clover is based on the `over_sampling` functions in the `imblearn` `module <https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.SMOTE.html#rd2827128d089-1>`_ in python, which extends the SMOTE algorithm to all classes, based on the sampling strategy. First, `Clover` instantiates the SMOTE method according to the independent variable types (continuous and/or categorical). Afterwards, if the dependent variable is categorical, the resampling strategy is `all` i.e. each class is oversampled proportionally to the original dataset. On the other hand, if the dependent variable is continuous, the library will add a fake majority class encompassing all the samples and a fake minority sample deleted at the end. The only hyperparameters that can be tuned with Optuna and Ray Tune is the number of neighbors. Once the generation process is finished, `Clover` will separate the real samples from the synthetic/oversampled ones.

.. code-block:: python

    """
    Wrapper of the SMOTE (Synthetic Minority Oversampling TEchnique) Python implementation from imbalanced-learn.

    :cvar name: the name of the metric
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param k_neighbors: the number of neighbors used to find the avatar
    """

    name = "SMOTE"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        k_neighbors: int = 5,
    ):

How to optimize the hyperparameters?
------------------------------------

The only hyperparameter of SMOTE is `k`, the number of the nearest neighbors to draw the new observation from. Optuna and Ray Tune, already implemented in the library, can be used to find the best value. For more details, please refer to the notebook `Tune hyperparameters <../../tutorial/tune_hyperparameters.nblink>`_ to learn how to use the optimizers.


References
----------

- `Synthesizing Electronic Health Records for Predictive Models in Low-Middle-Income Countries (LMICs) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10295936/>`_

- `Generating high-fidelity synthetic patient data for assessing machine learning healthcare software <https://www.nature.com/articles/s41746-020-00353-9>`_

- `SMOTE-ENC: A Novel SMOTE-Based Method to Generate Synthetic Data for Nominal and Continuous Features <https://www.mdpi.com/2571-5577/4/1/18>`_

- `RSMOTE: improving classification performance over imbalanced medical datasets <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7292850/>`_

- `Another use of SMOTE for interpretable data collaboration analysis <https://www.sciencedirect.com/science/article/pii/S0957417423008874>`_

- https://github.com/CRCHUM-CITADEL/clover/blob/main/generators/smote.py

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/synthetic_data_generation.ipynb

- https://github.com/CRCHUM-CITADEL/clover/blob/main/notebooks/tune_hyperparameters.ipynb
