# standard library
from typing import Dict, Tuple, Union

# 3rd party library
import pandas as pd
import numpy as np


class DPSmote:
    """
    Python implementation of differentially private SMOTE. Only continuous data is considered.

    See `Lut, Yuliia. Privacy-Aware Data Analysis: Recent Developments for Statistics and Machine Learning.
    Columbia University, 2022. <https://academiccommons.columbia.edu/doi/10.7916/he4k-zm64/download>`_
    for more details.

    :param l_connectivity: the distance to decide the neighborhood
    :param nu: granularity of the uniform grid that the data will be partitioned into
    :param r: each feature should fall into the range of [-r, r]
    :param epsilon: the privacy budget
    :param sampling_strategy: a dictionary to specify the number of samples to be generated for each target label
    :param random_state: for reproducibility purposes
    """

    def __init__(
        self,
        l_connectivity: int = 2,
        nu: float = None,
        r: float = None,
        epsilon: float = None,
        sampling_strategy: Dict[any, int] = None,
        random_state: int = None,
    ):
        self.l_connectivity = l_connectivity
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
            "l_connectivity": self.l_connectivity,
            "nu": self.nu,
            "r": self.r,
            "epsilon": self.epsilon,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

        return params

    def unifrom_grid_partition(
        self, df_X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partition data into equal-width cells.

        :param df_X: the data to be partitioned (all the variables should be continuous and falls into the same range)
        :return: the centers along 1 dimension, the centers of each cell, the count of data points in each cell
        """

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

        cell_centers = np.meshgrid(*grid_centers)
        cell_centers = np.vstack([center.flatten() for center in cell_centers]).T

        # Initiate the counts of data points in each cell
        counts = np.zeros((m,) * d, dtype=int)

        for _, row in df_X.iterrows():
            indices = tuple(
                int((coord + self.r) // (2 * nu_ * self.r))
                if int((coord + self.r) // (2 * nu_ * self.r)) < m
                else m - 1
                for coord in row
            )
            counts[indices] += 1

        return grid_centers[0], cell_centers, counts

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

            grid_centers_1d, _, counts = self.unifrom_grid_partition(X_idx)

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

            # Generate new data points one by one
            for _ in range(n_sample):
                new_data_point = []

                # Randomly select a data point
                chosen_idx_flat = np.random.choice(len(prob_flat), p=prob_flat)
                chosen_idx = np.unravel_index(chosen_idx_flat, prob.shape)

                # Find the neighbors of the selected point
                neighbors_index = []
                neighbors_counts = []
                for i in np.ndindex(prob.shape):
                    offset = np.array(chosen_idx) - np.array(i)
                    dist = np.sum(np.absolute(offset))

                    # Select the neighbors according to l connectivity
                    if dist > 0 and dist <= self.l_connectivity:
                        neighbors_index.append(i)
                        neighbors_counts.append(noisy_counts[i])

                if np.sum(np.array(neighbors_counts)) == 0:
                    raise ValueError(
                        "No neighbors are found. Please increase the l connectivity or decrease grid granularity."
                    )

                prob_neighbor = np.array(neighbors_counts) / np.sum(
                    np.array(neighbors_counts)
                )
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
