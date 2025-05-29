"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
"""

from sklearn.ensemble import RandomForestClassifier as RFC
from diffprivlib.models import RandomForestClassifier as DPRFC

from .base import ClassifierSampler


class RandomForestClassifier(ClassifierSampler):
    dp_clf_class = DPRFC
    clf_class = RFC
