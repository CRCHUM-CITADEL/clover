#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import Union, List, Tuple, Type  # standard library
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import pandas as pd  # 3rd party packages
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import utils.learning as ulearning  # local
import utils.draw as udraw
from metrics.utility.base import UtilityMetric
import metrics.utility.application as app


def get_metrics() -> List[Type[UtilityMetric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [Distinguishability, CrossRegression, CrossClassification]


class Distinguishability(UtilityMetric):
    """
    Check the similarity between the real and synthetic data by training a model to distinguish between them
    and measuring its performance.

    See `El Emam, Khaled, et al. "Utility Metrics for Evaluating Synthetic Health Data Generation Methods:
    Validation Study." JMIR medical informatics 10.4 (2022): e35734 <https://medinform.jmir.org/2022/4/e35734>`_
    for more details.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar min: the minimal bound
    :vartype min: Union[int, float]
    :cvar max: the maximal bound
    :vartype max: Union[int, float]
    :cvar objective: the target value for the metric: 'min' or 'max'
    :vartype objective: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
        for the prediction metrics only
    """

    name = "Distinguishability"
    alias = "dist"
    min = 0
    max = 1
    objective = "min"

    def __init__(
        self, random_state: int = None, num_repeat: int = 10, num_folds: int = 10
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._num_folds = num_folds

    @classmethod
    def get_average_submetrics(cls) -> List[str]:
        """
        Get the average submetrics of the current metric.

        :return: the list of the average submetrics
        """
        return [
            "propensity_mse",
            "prediction_mse_real",
            "prediction_mse_synth",
            "prediction_auc",
        ]

    @staticmethod
    def propensity_mse(propensity_scores: Union[List[float], np.ndarray]) -> float:
        """
        Compute the mean squared error between 0.5 (classifier cannot distinguish between real and synthetic data)
        and the predicted probabilities.

        :param propensity_scores: the predicted probabilities of being a real record
        :return: the propensity mean squared error
        """
        n = len(propensity_scores)
        pi = np.array(propensity_scores)

        pmse = 1 / n * np.sum((pi - 0.5) ** 2) / 0.25

        return pmse

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Compute three distinguishability metrics between real and synthetic datasets:
        the propensity mean squared error (all data), the prediction mean squared error (test only) and
        the prediction auc score (test only).

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions propensity mean squared error **propensity_mse** and
              the average across repetitions and folds prediction mean squared error for real and synthetic test sets
              **prediction_mse_real** and **prediction_mse_synth** and prediction auc score **prediction_auc**
            * **detailed** -- the propensity mean squared errors **propensity_mse**,
              the average across folds prediction mean squared errors for real and synthetic test sets
              **prediction_mse_real** and **prediction_mse_synth** and prediction auc scores **prediction_auc**,
              the predictions for real and synthetic test sets **prediction_real** and **prediction_synth**
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        # Transform original dataframes into input and output arrays for the training stage
        # TODO: ensure no need for imputation, normalization?
        df = pd.concat([df_real, df_synthetic], axis=0, ignore_index=True)

        #   Select the columns keeping the order
        cat_cols = [col for col in df_real.columns if col not in metadata["continuous"]]
        cont_cols = [col for col in df_real.columns if col not in cat_cols]
        df[cat_cols] = df[cat_cols].astype("object")

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), cont_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        drop="first",
                        categories=[df_real[cat].unique() for cat in cat_cols],
                    ),  # TODO: use infrequent categories?
                    cat_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

        #   Label 1 for real records and 0 for synthetic ones
        y = np.array([1] * len(df_real) + [0] * len(df_synthetic))

        # Compute the distinguishability score in three different settings
        dist_scores = []
        prediction_real = []
        prediction_synth = []

        # Compute scores several times to account for randomness
        for _ in range(self._num_repeat):
            # Propensity MSE - no train test split
            pipe = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("gbm", GradientBoostingClassifier()),
                ]
            )
            auc_score, y_pred_proba = ulearning.train_predict(
                pipeline=pipe,
                x_train=df,
                y_train=y,
                x_test_list=[df],
                y_test_list=[y],
                classif_labels=[0, 1],
            )
            propensity_score = self.propensity_mse(y_pred_proba[0])  # only one test set

            # Prediction MSE and AUC score on the test set with kfolds
            prediction_mse_real = []
            prediction_mse_synth = []
            prediction_auc = []

            if self._num_folds < 2:  # test data is equal to training data
                prediction_mse_real.append(propensity_score)
                prediction_mse_synth.append(propensity_score)
                prediction_auc.append(
                    max(0.5, auc_score[0]) * 2 - 1
                )  # scale between 0 and 1
            else:
                kf = KFold(n_splits=self._num_folds, shuffle=True)
                for tr_ind, te_ind in kf.split(df_real, y[: len(df_real)]):
                    # Add the synthetic indices and shuffle training indices
                    train_index = np.hstack([tr_ind, tr_ind + len(df_real)])
                    np.random.shuffle(train_index)
                    test_index = np.hstack([te_ind, te_ind + len(df_real)])

                    pipe = Pipeline(
                        steps=[
                            ("preprocessing", preprocessing),
                            ("gbm", GradientBoostingClassifier()),
                        ]
                    )

                    scores, y_pred_proba = ulearning.train_predict(
                        pipeline=pipe,
                        x_train=df.iloc[train_index],
                        y_train=y[train_index],
                        x_test_list=[df.iloc[test_index]],
                        y_test_list=[y[test_index]],
                        classif_labels=[0, 1],
                    )

                    y_pred_real = y_pred_proba[0][: len(test_index) // 2]
                    mse_real = self.propensity_mse(y_pred_real)  # only one test set
                    y_pred_synth = y_pred_proba[0][len(test_index) // 2 :]
                    mse_synth = self.propensity_mse(y_pred_synth)  # only one test set
                    auc = (
                        max(0.5, scores[0]) * 2 - 1
                    )  # only one test set and scale between 0 and 1

                    prediction_mse_real.append(mse_real)
                    prediction_mse_synth.append(mse_synth)
                    prediction_auc.append(auc)
                    prediction_real.extend(y_pred_real)
                    prediction_synth.extend(y_pred_synth)

            # Average scores on kfolds
            dist_scores.append(
                [
                    propensity_score,
                    np.mean(prediction_mse_real),
                    np.mean(prediction_mse_synth),
                    np.mean(prediction_auc),
                ]
            )

        dist_scores = np.array(dist_scores)

        # Average scores on repetitions
        (
            propensity_score,
            prediction_mse_real,
            prediction_mse_synth,
            prediction_auc,
        ) = np.mean(dist_scores, axis=0)

        res = {
            "average": {
                "propensity_mse": propensity_score,
                "prediction_mse_real": prediction_mse_real,
                "prediction_mse_synth": prediction_mse_synth,
                "prediction_auc": prediction_auc,
            },
            "detailed": {
                "propensity_mse": dist_scores[:, 0],
                "prediction_mse_real": dist_scores[:, 1],
                "prediction_mse_synth": dist_scores[:, 2],
                "prediction_auc": dist_scores[:, 3],
                "prediction_real": np.array(prediction_real),
                "prediction_synth": np.array(prediction_synth),
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the distinguishability scores and a boxplot to compare the predictions.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "propensity_mse",
                "prediction_mse_real",
                "prediction_mse_synth",
                "prediction_auc",
                "prediction_real",
                "prediction_synth",
            ]
        )

        # Bar plot single value
        plt.figure(figsize=figsize, layout="constrained")

        data = pd.DataFrame(
            {
                "propensity_mse": report["propensity_mse"],
                "prediction_mse_real": report["prediction_mse_real"],
                "prediction_mse_synth": report["prediction_mse_synth"],
                "prediction_auc": report["prediction_auc"],
            }
        )
        udraw.bar_plot(
            data=data,
            title=f"Metric: {cls.name}",
            value_name="Distinguishability score",
        )

        # Box plot for predictions
        plt.figure(figsize=figsize, layout="constrained")

        data = pd.DataFrame(
            {
                "prediction_real": report["prediction_real"],
                "prediction_synth": report["prediction_synth"],
            }
        )

        udraw.box_plot(
            data=data,
            title=f"Metric: {cls.name}",
        )


class CrossLearning(UtilityMetric, metaclass=ABCMeta):
    """
    Check the preservation of all the relationship between the variables by generating predictions
    for a variable based on the others.

    The method was adapted from the article `Goncalves, Andre, et al.
    "Generation and evaluation of synthetic patient data." BMC medical research methodology 20.1 (2020): 1-40.
    <https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1>`_

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar min: the minimal bound
    :vartype min: Union[int, float]
    :cvar max: the maximal bound
    :vartype max: Union[int, float]
    :cvar objective: the target value for the metric: 'min' or 'max'
    :vartype objective: str
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = "Cross"
    alias = "cross"
    min = 0
    objective = "min"
    class_name: str

    @classmethod
    @property
    @abstractmethod
    def class_name(cls) -> str:
        """
        :return: the name of the class called by the metric to train a predictor
        """

    def __init__(
        self,
        random_state: int = None,
        num_repeat: int = 10,
        num_folds: int = 10,
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._num_folds = num_folds

    @classmethod
    def get_average_submetrics(cls) -> List[str]:
        """
        Get the average submetrics of the current metric.

        :return: the list of the average submetrics
        """
        return ["real_train", "synth_train"]

    def compute(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        metadata: dict,
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset or the synthetic one.

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across all variables to predict of the absolute difference between
              the real test set and the synthetic test set scores:
              **real_train** when the model is trained on the real data and
              **synth_train** when trained on the synthetic data
            * **detailed** -- four dictionaries with the **dependent_vars** as keys and the scores
              as values: **real_real** and **real_synth** when the model is trained on
              the real data and tested on real and synthetic holdouts respectively, **synth_real** and
              **synth_synth** when trained on the synthetic data and tested on real and
              synthetic holdouts sets respectively

            or *empty* if there is no **dependent_vars** to predict
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        dependent_vars = (
            metadata["continuous"]
            if self.__class__.class_name == "Regression"
            else metadata["categorical"]
        )

        if len(dependent_vars) == 0:
            return {}
        if df_real.shape[1] <= 1:
            return {}

        dic_realtrain_realtest = {}
        dic_realtrain_synthtest = {}
        dic_synthtrain_realtest = {}
        dic_synthtrain_synthtest = {}
        diff_realtrain = []
        diff_synthtrain = []

        for col in dependent_vars:
            metadata_pred = deepcopy(metadata)
            metadata_pred["variable_to_predict"] = col

            pred = getattr(app, self.__class__.class_name)(
                num_repeat=self._num_repeat, num_folds=self._num_folds
            )
            res = pred.compute(df_real, df_synthetic, metadata_pred)
            if len(res) == 0:
                continue
            else:
                res = res["detailed"]

            dic_realtrain_realtest[col] = res["score_real_real"]
            dic_realtrain_synthtest[col] = res["score_real_synth"]
            dic_synthtrain_synthtest[col] = res["score_synth_synth"]
            dic_synthtrain_realtest[col] = res["score_synth_real"]

            # Absolute difference for the average report
            diff_realtrain.append(
                abs(dic_realtrain_realtest[col] - dic_realtrain_synthtest[col])
            )
            diff_synthtrain.append(
                abs(dic_synthtrain_realtest[col] - dic_synthtrain_synthtest[col])
            )

        if len(dic_realtrain_realtest) == 0:
            return {}

        res = {
            "average": {
                "real_train": np.mean(diff_realtrain),
                "synth_train": np.mean(diff_synthtrain),
            },
            "detailed": {
                "real_real": dic_realtrain_realtest,
                "real_synth": dic_realtrain_synthtest,
                "synth_real": dic_synthtrain_realtest,
                "synth_synth": dic_synthtrain_synthtest,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the real and synthetic test sets predictions
        when the model is trained on the real dataset or the synthetic one.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert all(
            key in report
            for key in [
                "real_real",
                "real_synth",
                "synth_real",
                "synth_synth",
            ]
        )

        # One plot per variable
        num_cols = len(report["real_real"])
        max_plot_per_window = 6
        num_win = num_cols // max_plot_per_window
        if num_cols % max_plot_per_window != 0:
            num_win += 1

        axes = []
        for f in range(num_win):
            fig, axes_f = plt.subplots(
                ncols=max_plot_per_window, figsize=figsize, layout="constrained"
            )
            fig.suptitle(f"Metric: {cls.name} ({f + 1}/{num_win})")
            axes.extend(axes_f)

        for i, col in enumerate(report["real_real"]):
            df = pd.DataFrame(
                np.column_stack((report["real_real"][col], report["synth_real"][col])),
                columns=["Real", "Synthetic"],
            )
            df_nested = pd.DataFrame(
                np.column_stack(
                    (report["real_synth"][col], report["synth_synth"][col])
                ),
                columns=["Real", "Synthetic"],
            )

            udraw.bar_plot_per_column_hue(
                df=df,
                df_nested=df_nested,
                original_name="Real",
                nested_name="Synthetic",
                hue_name="Tested on",
                orient="v",
                title=col,
                order=["Real", "Synthetic"],
                value_name=getattr(app, cls.class_name).score_name,
                xrotation=True,
                ax=axes[i],
            )

            axes[i].set_xlabel("Trained on")
            if i % max_plot_per_window != 0:
                axes[i].get_legend().remove()


class CrossRegression(CrossLearning):
    """
    Check the preservation of all the relationship between the variables by generating predictions
    for a continuous variable based on the others.

    The method was adapted from the article `Goncalves, Andre, et al.
    "Generation and evaluation of synthetic patient data." BMC medical research methodology 20.1 (2020): 1-40.
    <https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1>`_

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar min: the minimal bound
    :vartype min: Union[int, float]
    :cvar max: the maximal bound
    :vartype max: Union[int, float]
    :cvar objective: the target value for the metric: 'min' or 'max'
    :vartype objective: str
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = CrossLearning.name + " Regression"
    alias = CrossLearning.alias + "_reg"
    max = np.inf
    class_name = "Regression"


class CrossClassification(CrossLearning):
    """
    Check the preservation of all the relationship between the variables by generating predictions
    for a categorical variable based on the others.

    The method was adapted from the article `Goncalves, Andre, et al.
    "Generation and evaluation of synthetic patient data." BMC medical research methodology 20.1 (2020): 1-40.
    <https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1>`_

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar min: the minimal bound
    :vartype min: Union[int, float]
    :cvar max: the maximal bound
    :vartype max: Union[int, float]
    :cvar objective: the target value for the metric: 'min' or 'max'
    :vartype objective: str
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = CrossLearning.name + " Classification"
    alias = CrossLearning.alias + "_classif"
    max = 1
    class_name = "Classification"
