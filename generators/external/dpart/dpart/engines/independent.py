"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from .. import dpart
from ..methods import ProbabilityTensor


class Independent(dpart):
    def __init__(self, bounds: dict = None, n_bins: int = 20, epsilon: dict = None):
        self.n_bins = n_bins
        super().__init__(epsilon=epsilon, bounds=bounds, n_parents=0)

    def default_method(self, dtype):
        return ProbabilityTensor(n_bins=self.n_bins)
