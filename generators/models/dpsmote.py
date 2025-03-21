# standard library
from typing import Dict, Tuple, Union

# 3rd party library
import pandas as pd
import numpy as np

# import itertools
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class DPSmote:
    """
    Python implementation of differentially private SMOTE. Only continuous data is considered.

    Adapted and modified from `Lut, Yuliia. Privacy-Aware Data Analysis: Recent Developments for Statistics and Machine Learning.
    Columbia University, 2022. <https://academiccommons.columbia.edu/doi/10.7916/he4k-zm64/download>`_
    for more details.

    :param k_neighbors: the number of neighbors used to find the avatar
    :param nu: granularity of the uniform grid that the data will be partitioned into
    :param r: each feature should fall into the range of [-r, r]
    :param epsilon: the privacy budget
    :param sampling_strategy: a dictionary to specify the number of samples to be generated for each target label
    :param random_state: for reproducibility purposes
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        nu: float = None,
        r: float = None,
        epsilon: float = None,
        sampling_strategy: Dict[any, int] = None,
        random_state: int = None,
    ):
        self.k_neighbors = k_neighbors
        self.nu = nu
        self.r = r
        self.epsilon = epsilon
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def set_params(self, **kwargs) -> None:
        """
        Set the parameter of the configuration.

        :param: the parameter and the value to be set/reset
        :return: *None*
        """

        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise AttributeError(f"{param} is not a valid parameter")

    def get_params(self) -> dict:
        """
        Get the configuration of DP-SMOTE.

        :return: the parameters for the configuration
        """
        params = {
            "k_neighbors": self.k_neighbors,
            "nu": self.nu,
            "r": self.r,
            "epsilon": self.epsilon,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

        return params

    def unifrom_grid_partition(
        self, df_X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partition data into equal-width cells.

        :param df_X: the data to be partitioned (all the variables should be continuous and falls into the same range)
        :return: the centers along 1 dimension and the count of data points in each cell
        """

        # def generate_cell_centers(grid_centers):
        #     # Cartesian product of grid centers along all dimensions
        #     cell_centers = np.array(list(itertools.product(*grid_centers)))
        #     return cell_centers

        d = df_X.shape[1]  # The dimension of the grid/cell
        m = int(1 // self.nu)  # Number of partitions along each dimension
        nu_ = 1 / m

        grid_centers = []
        for i in range(d):
            grid_centers.append(
                np.linspace(
                    start=-self.r + self.r * nu_, stop=self.r - self.r * nu_, num=m
                )
            )

        # cell_centers = generate_cell_centers(grid_centers)
        # cell_centers = np.vstack([center.flatten() for center in cell_centers]).T

        # Initiate the counts of data points in each cell
        counts = np.zeros((m,) * d, dtype=int)

        for _, row in df_X.iterrows():
            indices = tuple(
                (
                    int((coord + self.r) // (2 * nu_ * self.r))
                    if int((coord + self.r) // (2 * nu_ * self.r)) < m
                    else m - 1
                )
                for coord in row
            )
            counts[indices] += 1

        return grid_centers[0], counts  # cell_centers

    def fit_resample(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Generate new samples with differentially private SMOTE

        :param X: the input data (features)
        :param y: the target
        :return: synthetic data with target appended
        """

        y = np.array(y)

        synthetic_data = []

        # Generate new samples for each target label
        for label, n_sample in self.sampling_strategy.items():
            # Initiate the collection of data for a specific label
            synthetic_data_label = []

            idx = np.where(y == label)[0]

            X_idx = X.iloc[idx, :]

            grid_centers_1d, counts = self.unifrom_grid_partition(X_idx)

            if self.epsilon is not None:
                noisy_counts = counts + np.random.laplace(
                    0, scale=1 / self.epsilon, size=counts.shape
                )

                noisy_counts[noisy_counts < 0] = 0  # Set negative value to 0
            else:
                noisy_counts = counts  # Non-private

            # Convert counts to probability
            prob = noisy_counts / np.sum(noisy_counts)
            prob_flat = prob.flatten()

            # Compute index of all the data points
            idx_all = (
                np.indices(prob.shape).reshape(len(prob.shape), -1).T.astype(np.int8)
            )

            nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, n_jobs=-1)
            nn.fit(idx_all)

            # Generate new data points one by one
            for _ in tqdm(range(n_sample), desc=f"Label = {label}"):
                new_data_point = []

                # Initiate the total count in the neighboring cells
                total_count = 0

                while total_count == 0:
                    # Randomly select a data point
                    chosen_idx_flat = np.random.choice(len(prob_flat), p=prob_flat)
                    chosen_idx = np.unravel_index(chosen_idx_flat, prob.shape)

                    ##########################################
                    # Find the neighbors of the selected point
                    ##########################################

                    # Find the k-nearest neighbors
                    dist, neighbors = nn.kneighbors(
                        np.array(chosen_idx).reshape(1, -1), return_distance=True
                    )
                    idx_valid = neighbors[0][np.where(dist[0] != 0)]

                    neighbors_index = [tuple(i) for i in idx_all[idx_valid]]
                    neighbors_counts = [noisy_counts[i] for i in neighbors_index]

                    total_count = np.sum(np.array(neighbors_counts))

                prob_neighbor = np.array(neighbors_counts) / total_count
                chosen_neighbor_idx = neighbors_index[
                    np.random.choice(len(neighbors_index), p=prob_neighbor)
                ]

                # Generate value for each dimension of the new data point
                for i, idx_1d in enumerate(chosen_neighbor_idx):
                    u = np.random.uniform()
                    z = grid_centers_1d[chosen_idx[i]] + u * (
                        grid_centers_1d[idx_1d] - grid_centers_1d[chosen_idx[i]]
                    )

                    new_data_point.append(z)

                synthetic_data_label.append(new_data_point)

            df_synth_label = pd.DataFrame(synthetic_data_label, columns=X.columns)
            df_synth_label["Target"] = label

            synthetic_data.append(df_synth_label)

        df_synthetic_data = pd.concat(synthetic_data, ignore_index=True)

        return df_synthetic_data
