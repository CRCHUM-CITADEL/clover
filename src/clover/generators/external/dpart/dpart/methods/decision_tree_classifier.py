"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from sklearn.tree import DecisionTreeClassifier as DTC
from diffprivlib.models.forest import DecisionTreeClassifier as DPDTC

from .base import ClassifierSampler


class DecisionTreeClassifier(ClassifierSampler):
    dp_clf_class = DPDTC
    clf_class = DTC
