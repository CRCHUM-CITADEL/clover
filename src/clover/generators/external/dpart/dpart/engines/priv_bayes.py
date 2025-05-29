"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from .. import dpart
from ..methods import ProbabilityTensor


class PrivBayes(dpart):
    def __init__(
        self,
        bounds: dict = None,
        n_parents: int = 2,
        n_bins: int = 20,
        epsilon: dict = None,
    ):
        self.n_bins = n_bins
        super().__init__(
            epsilon=epsilon, prediction_matrix="infer", n_parents=n_parents
        )

    def default_method(self, dtype):
        return ProbabilityTensor(n_parents=None, n_bins=self.n_bins)
