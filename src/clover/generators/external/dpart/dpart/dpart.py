"""
This file is under the following license and copyright.
MIT License
Copyright (c) 2022 dpart

The following modifications were made to the file:
    - The paths of imported modules were modified to be relative.
    - The lambda function was replaced by a named function so that the object can be saved with pickle.
    - The feature_range of the MinMaxScaler was changed from a list into a tuple to match the type required by sklearn.
    - Bug was fixed by converting the type of the columns to str type when generating synthetic data.
    - The privacy budget accountant was implemented to track privacy spend.
    - Maximum number of iterations was added to fix the convergence issue.
    - Warning for using default method (when not specified) was removed.
    - Warnings from DP-models were removed.
    - Typo was fixed in warning message for privacy leakage of categorical variables.
    - Unshown warning message was fixed for bounds of continuous variables.
    - Privacy leakage for decoding predictions was fixed.
    - Tree models were added.
    - max_depth parameter was added for tree model.
"""

import warnings
import numpy as np
import pandas as pd
from logging import getLogger
from typing import Union
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from diffprivlib import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

from .utils.dependencies import DependencyManager
from .methods import (
    ProbabilityTensor,
    LinearRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
)


logger = getLogger("dpart")
logger.setLevel("ERROR")


class dpart:
    default_numerical = ProbabilityTensor
    default_categorical = ProbabilityTensor

    def __init__(
        self,
        # methods
        methods: dict = None,
        # privacy settings
        epsilon: Union[dict, float] = None,
        bounds: dict = None,
        slack: float = 0.0,
        # dependencies
        visit_order: list = None,
        prediction_matrix: dict = None,
        n_parents=None,
        **kwargs,
    ):
        # Private budget
        if epsilon is not None:
            if not isinstance(epsilon, dict):
                if prediction_matrix == "infer":
                    epsilon = {"dependency": epsilon / 2, "methods": epsilon / 2}
                else:
                    epsilon = {"dependency": 0, "methods": epsilon}
        else:
            epsilon = {"dependency": None, "methods": {}}
        self._epsilon = epsilon
        self.slack = slack
        self.params = kwargs
        self.dep_manager = DependencyManager(
            epsilon=self._epsilon.get("dependency", None),
            visit_order=visit_order,
            prediction_matrix=prediction_matrix,
            n_parents=n_parents,
        )

        # method dict
        if methods is None:
            methods = {}
        self.methods = methods
        self.encoders = None

        # bound dict
        if bounds is None:
            bounds = {}
        self.bounds = bounds
        self.dtypes = None
        self.root = None
        self.columns = None

    def root_column(self, df: pd.DataFrame) -> str:
        root_col = "__ROOT__"
        idx = 0
        while root_col in df.columns:
            root_col = f"__ROOT_{idx}__"
            idx += 1
        return root_col

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        self.encoders = {}
        df = df.copy()

        def is_numeric(x):
            return isinstance(x, (np.floating, np.integer, int, float))

        for col, series in df.items():
            if series.dtype.kind in "OSb":
                t_dtype = "category"
                if col not in self.bounds:
                    if self._epsilon.get("methods", None) is not None:
                        warnings.warn(
                            f"List of categories not specified for column '{col}'",
                            PrivacyLeakWarning,
                        )

                    if pd.Series(series.unique()).apply(is_numeric).astype(bool).all():
                        self.bounds[col] = {"categories": sorted(list(series.unique()))}
                    else:
                        self.bounds[col] = {
                            "categories": sorted(list(series.unique()), key=str)
                        }
                self.encoders[col] = OrdinalEncoder(
                    categories=[self.bounds[col]["categories"]]
                )
            else:
                t_dtype = "float"
                if col not in self.bounds:
                    if self._epsilon.get("methods", None) is not None:
                        warnings.warn(
                            f"upper and lower bounds not specified for column '{col}'",
                            PrivacyLeakWarning,
                        )
                    self.bounds[col] = {"min": series.min(), "max": series.max()}
                self.encoders[col] = MinMaxScaler(
                    feature_range=(self.bounds[col]["min"], self.bounds[col]["max"])
                )
            df[col] = pd.Series(
                self.encoders[col].fit_transform(df[[col]]).squeeze(),
                name=col,
                index=df.index,
                dtype=t_dtype,
            )

        return df

    def default_method(self, dtype):
        if dtype.kind in "OSb":
            if self.default_categorical == LogisticRegression:
                return self.default_categorical(
                    max_iter=self.params.get("max_iter", 10000)
                )
            elif self.default_categorical == DecisionTreeClassifier:
                return self.default_categorical(
                    max_depth=self.params.get("max_depth", 5)
                )
            elif self.default_categorical == RandomForestClassifier:
                return self.default_categorical(
                    max_depth=self.params.get("max_depth", 5)
                )
            else:
                return self.default_categorical()
        return self.default_numerical()

    def fit(self, df: pd.DataFrame):
        # dependency manager
        t_df = self.dep_manager.preprocess(df)
        self.dep_manager.fit(t_df)

        # Capture dtypes
        self.dtypes = df.dtypes
        self.columns = df.columns

        if not isinstance(self._epsilon["methods"], dict):
            col_budget = float(self._epsilon["methods"]) / df.shape[1]
            self._epsilon["methods"] = {col: col_budget for col in self.columns}

        # reorder and introduce initial columns
        self.root = self.root_column(df)
        t_df = self.normalise(df)
        t_df.insert(0, column=self.root, value=0)

        # initiate budget accountant
        if self._epsilon.get("dependency", None) in [None, 0]:
            spent_budget = None
        else:
            spent_budget = [(self._epsilon["dependency"], 0)]
        self.budget_acc = BudgetAccountant(slack=self.slack, spent_budget=spent_budget)
        self.budget_acc.set_default()

        # build methods
        for idx, target in enumerate(self.dep_manager.visit_order):
            X_columns = [self.root] + self.dep_manager.prediction_matrix.get(target, [])
            X = t_df[X_columns]
            y = t_df[target]

            if y.nunique() < 2:
                warnings.warn(
                    f"target {target} is static method will default to ProbabilityTensor."
                )
                self.methods[target] = ProbabilityTensor()
            elif target not in self.methods:
                def_method = self.default_method(self.dtypes[target])
                # warnings.warn(
                #     f"target {target} has no specified method will use default {def_method.__class__.__name__}"
                # )
                self.methods[target] = def_method

            if self._epsilon["methods"].get(target, None) is not None:
                self.methods[target].set_epsilon(self._epsilon["methods"][target])

                # update the budget spent, if the method is linear regression
                if isinstance(self.methods[target], LinearRegression):
                    self.budget_acc.spend(
                        epsilon=self._epsilon["methods"][target] / 2, delta=0
                    )

            logger.info(
                f"Fit target: {target} | sampler used: {self.methods[target].__class__.__name__}"
            )

            t_X, t_y = self.methods[target].preprocess(X=X, y=y)

            # Supress warnings from DP-models
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.methods[target].fit(X=t_X, y=t_y)

    def denormalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if isinstance(
                self.encoders[col], OrdinalEncoder
            ):  # Only decode categorical variables
                df[col] = self.encoders[col].inverse_transform(df[[col]]).squeeze()

            if self.dtypes[col].kind in "ui":
                df[col] = df[col].round().astype(int).astype(self.dtypes[col])
            else:
                df[col] = df[col].astype(self.dtypes[col])
        return df

    def sample(self, n_records: int) -> pd.DataFrame:
        df = pd.DataFrame({self.root: 0}, index=np.arange(n_records))
        for target in self.dep_manager.visit_order:
            X_columns = [self.root] + self.dep_manager.prediction_matrix.get(target, [])
            logger.info(f"Sample target {target}")
            logger.debug(f"Sample target {target} - preprocess feature matrix")
            #### Convert a mix of str and np.str_ to str type ####
            df_features = df[X_columns]
            df_features.columns = df_features.columns.astype(str)
            ######################################################
            t_X = self.methods[target].preprocess_X(df_features)
            logger.debug(f"Sample target {target} - Sample values")
            t_y = self.methods[target].sample(X=t_X)
            logger.debug(f"Sample target {target} - post process sampled values")
            y = self.methods[target].postprocess_y(y=t_y)
            logger.debug(f"Sample target {target} - Update feature matrix")
            df.insert(loc=df.shape[1], column=target, value=y)

        logger.info("denormalise sampled data")
        i_df = self.denormalise(df=df.drop(self.root, axis=1)).reindex(
            columns=self.columns
        )
        return i_df

    @property
    def epsilon(self):
        budgets = [method.epsilon for _, method in self.methods.items()] + [
            self.dep_manager.epsilon
        ]
        if pd.isnull(budgets).any():
            return None
        else:
            return sum(budgets)
