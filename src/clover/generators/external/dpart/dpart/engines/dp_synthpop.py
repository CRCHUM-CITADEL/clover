"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
    - The differential private synthpop class was modified to include more arguments.
    - The type hints were updated.
    - Tree model was used as default model.
"""

from typing import Dict, Union, List
from .. import dpart
from ..methods import RandomForestClassifier, LinearRegression


class DPSynthpop(dpart):
    default_numerical = LinearRegression
    default_categorical = RandomForestClassifier

    def __init__(
        self,
        methods: dict = None,
        epsilon: Union[float, Dict[str, Union[float, Dict[str, float]]]] = None,
        bounds: Dict[str, Union[List, Dict[str, List]]] = None,
        slack: float = 0.0,
        visit_order: List[str] = None,
        prediction_matrix: Union[str, Dict[str, List[str]]] = None,
        n_parents: int = None,
        **kwargs,
    ):
        super().__init__(
            methods=methods,
            epsilon=epsilon,
            bounds=bounds,
            slack=slack,
            visit_order=visit_order,
            prediction_matrix=prediction_matrix,
            n_parents=n_parents,
            **kwargs,
        )
