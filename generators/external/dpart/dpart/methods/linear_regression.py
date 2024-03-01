"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from sklearn.linear_model import LinearRegression as LR
from diffprivlib.models import LinearRegression as DPLR

from .base import RegressorSampler


class LinearRegression(RegressorSampler):
    dp_reg_class = DPLR
    reg_class = LR
