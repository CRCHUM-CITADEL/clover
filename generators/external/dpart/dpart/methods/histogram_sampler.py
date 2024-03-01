"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from .probability_tensor import ProbabilityTensor


class HistogramSampler(ProbabilityTensor):
    def __init__(self, epsilon: float = None, n_bins=20):
        super().__init__(epsilon=epsilon, n_bins=n_bins, n_parents=0)
