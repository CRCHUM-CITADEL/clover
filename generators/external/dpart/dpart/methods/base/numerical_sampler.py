"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from .sampler import Sampler


class NumericalSampler(Sampler):
    category_support: bool = False
    numerical_support: bool = True
