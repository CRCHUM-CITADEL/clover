"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from sklearn.linear_model import LogisticRegression as LR
from diffprivlib.models import LogisticRegression as DPLR

from .base import ClassifierSampler


class LogisticRegression(ClassifierSampler):
    dp_clf_class = DPLR
    clf_class = LR
