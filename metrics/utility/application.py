#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import List, Tuple, Type  # standard library
from abc import ABCMeta, abstractmethod
import warnings

import pandas as pd  # 3rd party packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import utils.draw as udraw  # local
import utils.learning as ulearning
from metrics.utility.base import UtilityMetric


def get_metrics() -> List[Type[UtilityMetric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [Regression, Classification, FScore]


class Prediction(UtilityMetric, metaclass=ABCMeta):
    """
    Check that the synthetic data have the same behavior as the real data regarding the application task.

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
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = "Prediction"
    alias = "prediction"
    min = 0
    max = np.inf
    objective = "min"
    score_name: str

    @classmethod
    @property
    @abstractmethod
    def score_name(cls) -> str:
        """
        :return: the name of the score computed by the metric
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
        return ["diff_real_train", "diff_synth_train", "diff_real_synth"]

    @staticmethod
    @abstractmethod
    def _learning(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target: pd.Series,
        num_folds: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a classifier or a regressor and score the predictions for the test sets. To be reimplemented.

        :param x_train: the training inputs
        :param y_train: the training ground truth
        :param x_test: the test input
        :param y_test: the test ground truth
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: the average scores for **x_train** and **x_test** on k-folds
        """
        pass

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference between the real and the synthetic performances
            * **detailed** -- a dictionary containing the scores for real and synthetic test datasets
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        var_pred = metadata["variable_to_predict"]
        if var_pred is None:
            return {}
        if df_real.shape[1] <= 1:
            return {}

        # Create x and y data
        df_real_x = df_real.drop(columns=var_pred)
        df_synthetic_x = df_synthetic.drop(columns=var_pred)
        y_real = df_real[var_pred].to_numpy()
        y_synth = df_synthetic[var_pred].to_numpy()

        # Transform categorical columns in one-hot format
        #   Ensure the output is binary if there are two classes
        if var_pred in metadata["categorical"] and df_real[var_pred].nunique() == 2:
            lb = LabelBinarizer()
            y = np.concatenate([y_real, y_synth])
            lb.fit(y)
            y_real = lb.transform(y_real).flatten()
            y_synth = lb.transform(y_synth).flatten()

        #   Select the categorical columns to transform
        cat_cols = list(set(metadata["categorical"]) - {var_pred})
        other_cols = list(set(df_real.columns) - set(cat_cols) - {var_pred})

        #   One-hot conversion and transformation to numpy arrays of the input
        encoder = OneHotEncoder(drop="first")
        df_x = pd.concat([df_real_x, df_synthetic_x], axis=0, ignore_index=True)
        encoder.fit(df_x[cat_cols])

        x_real_cat = encoder.transform(X=df_real_x[cat_cols]).toarray()
        x_real = np.concatenate([df_real_x[other_cols].to_numpy(), x_real_cat], axis=1)
        x_synth_cat = encoder.transform(X=df_synthetic_x[cat_cols]).toarray()
        x_synth = np.concatenate(
            [df_synthetic_x[other_cols].to_numpy(), x_synth_cat], axis=1
        )

        # Compute the cross learning in both directions
        scores_real_real = []
        scores_real_synth = []
        scores_synth_synth = []
        scores_synth_real = []

        # Compute scores several times to account for randomness
        for i in range(self._num_repeat):
            # Prediction MSE and AUC score on the test set with kfolds
            state = np.random.get_state()
            score_real_real, score_real_synth = self._learning(
                x_train=x_real,
                y_train=y_real,
                x_test=x_synth,
                y_test=y_synth,
                target=df_real[var_pred],
                num_folds=self._num_folds,
            )

            np.random.set_state(
                state
            )  # ensure that the results are identical for the same datasets
            score_synth_synth, score_synth_real = self._learning(
                x_train=x_synth,
                y_train=y_synth,
                x_test=x_real,
                y_test=y_real,
                target=df_real[var_pred],
                num_folds=self._num_folds,
            )

            scores_real_real.append(score_real_real)
            scores_real_synth.append(score_real_synth)
            scores_synth_synth.append(score_synth_synth)
            scores_synth_real.append(score_synth_real)

        diff_real_train = abs(np.array(scores_real_real) - np.array(scores_real_synth))
        diff_synth_train = abs(
            np.array(scores_synth_real) - np.array(scores_synth_synth)
        )
        diff_real_synth = abs(np.array(scores_real_real) - np.array(scores_synth_synth))

        res = {
            "average": {
                "diff_real_train": np.mean(diff_real_train),
                "diff_synth_train": np.mean(diff_synth_train),
                "diff_real_synth": np.mean(diff_real_synth),
            },
            "detailed": {
                "score_real_real": np.array(scores_real_real),
                "score_real_synth": np.array(scores_real_synth),
                "score_synth_real": np.array(scores_synth_real),
                "score_synth_synth": np.array(scores_synth_synth),
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the real and synthetic test sets predictions.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert all(key in report for key in ["score_real_real", "score_synth_synth"])

        plt.figure(figsize=figsize, layout="constrained")

        df = pd.DataFrame(
            np.column_stack(
                (
                    report["score_real_real"],
                    report["score_synth_synth"],
                    report["score_real_synth"],
                    report["score_synth_real"],
                )
            ),
            columns=[
                "Trained real test real",
                "Trained synthetic test synthetic",
                "Trained real test synthetic",
                "Trained synthetic test real",
            ],
        )

        udraw.bar_plot(
            data=df,
            title=f"Metric: {cls.name}",
            value_name=f"{cls.score_name}",
            orient="v",
        )


class Regression(Prediction):
    """
    Check that the synthetic data have the same behavior as the real data when performing a regression task.

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
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = "Regression"
    alias = "regression"
    score_name = "Mean Squared Error"

    @staticmethod
    def _learning(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target: pd.Series,
        num_folds: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a regressor and score the predictions for the test sets.

        :param x_train: the training inputs
        :param y_train: the training ground truth
        :param x_test: the test input
        :param y_test: the test ground truth
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: the average scores for **x_train** and **x_test** on k-folds
        """

        mse_train = []
        mse_test = []
        kf = KFold(n_splits=num_folds, shuffle=True)
        for train_index, test_index in kf.split(x_train, y_train):
            pipe = Pipeline(
                steps=[
                    ("standardization", StandardScaler()),
                    ("gbm", GradientBoostingRegressor()),
                ]
            )
            scores, _ = ulearning.train_predict(
                pipeline=pipe,
                x_train=x_train[train_index],
                y_train=y_train[train_index],
                x_test_list=[x_train[test_index], x_test[test_index]],
                y_test_list=[y_train[test_index], y_test[test_index]],
            )
            mse_train.append(scores[0])
            mse_test.append(scores[1])

        return np.mean(mse_train), np.mean(mse_test)

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference between the real and the synthetic performances
            * **detailed** -- a dictionary containing the scores for real and synthetic test datasets
        """

        if (
            metadata["variable_to_predict"] is None
            or metadata["variable_to_predict"] in metadata["categorical"]
        ):
            return {}

        res = super().compute(df_real, df_synthetic, metadata)

        return res


class Classification(Prediction):
    """
    Check that the synthetic data have the same behavior as the real data when performing a classification task.

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
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_folds: the scores are averaged across the number of folds to account for split randomness
    """

    name = "Classification"
    alias = "classif"
    max = 1
    score_name = "AUC score"

    @staticmethod
    def _learning(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target: pd.Series,
        num_folds: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a classifier and score the predictions for the test sets.

        :param x_train: the training inputs
        :param y_train: the training ground truth
        :param x_test: the test input
        :param y_test: the test ground truth
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: the average scores for **x_train** and **x_test** on k-folds
        """

        labels = target.unique()
        labels.sort()

        auc_train = []
        auc_test = []
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        for train_index, test_index in kf.split(x_train, y_train):
            pipe = Pipeline(
                steps=[
                    ("standardization", StandardScaler()),
                    ("gbm", GradientBoostingClassifier()),
                ]
            )
            scores, _ = ulearning.train_predict(
                pipeline=pipe,
                x_train=x_train[train_index],
                y_train=y_train[train_index],
                x_test_list=[x_train[test_index], x_test[test_index]],
                y_test_list=[y_train[test_index], y_test[test_index]],
                classif_labels=labels,
            )
            auc_train.append(scores[0])
            auc_test.append(scores[1])

        return np.mean(auc_train), np.mean(auc_test)

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference between the real and the synthetic performances
            * **detailed** -- a dictionary containing the scores for real and synthetic test datasets
        """

        if (
            metadata["variable_to_predict"] is None
            or metadata["variable_to_predict"] in metadata["continuous"]
        ):
            return {}

        if set(df_real[metadata["variable_to_predict"]].unique()) != set(
            df_synthetic[metadata["variable_to_predict"]].unique()
        ):
            warnings.warn(
                message=f"The datasets do not have the same labels for the variable {metadata['variable_to_predict']}. "
                f"The metric {self.name} cannot be computed.",
                category=UserWarning,
            )
            return {}

        return super().compute(df_real, df_synthetic, metadata)


class FScore(UtilityMetric):
    """
    Check the similarities of the F-scores for each feature of the real and synthetic datasets.

    The F-score is a feature selection technique to evaluate the discrimination potential of a feature.

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
    """

    name = "FScore"
    alias = "fscore"
    min = 0
    max = np.inf  # TODO: is it bounded?
    objective = "min"

    @classmethod
    def get_average_submetrics(cls) -> List[str]:
        """
        Get the average submetrics of the current metric.

        :return: the list of the average submetrics
        """
        return ["diff_f_score"]

    @staticmethod
    def fscore(df: pd.DataFrame, predicted_var: str) -> pd.Series:
        """
        Compute the F-Scores.

        See `Chen, Y. W., & Lin, C. J. (2006). Combining SVMs with various feature selection strategies.
        Feature extraction: foundations and applications, 315-324.
        <https://link.springer.com/chapter/10.1007/978-3-540-35488-8_13>`_ for more details.

        :param df: the dataframe containing the continuous variables to discriminate and the **predicted_var**
        :param predicted_var: the binary variable that will be predicted
        :return: the F-scores for all continuous variables
        """

        assert isinstance(
            df, pd.DataFrame
        ), "The input data should be a pandas dataframe"
        assert (
            len(df.columns) >= 2
        ), "The dataset is required to have at least one feature and one dependent variable"
        assert (
            predicted_var in df.columns
        ), "The dependent variable should be in the dataset"
        assert set(df[predicted_var].unique()) == {
            0,
            1,
        }, "The dependent variable should be binary"

        independent_vars = list(set(df.columns) - {predicted_var})
        counts = df[predicted_var].value_counts()

        df_0 = df.loc[df[predicted_var] == 0, independent_vars]
        df_1 = df.loc[df[predicted_var] == 1, independent_vars]
        df_all = df.loc[:, independent_vars]

        mean_all = df_all.mean(axis=0)
        mean_0 = df_0.mean(axis=0)
        mean_1 = df_1.mean(axis=0)

        sum_df_0 = ((df_0 - mean_0) ** 2).sum(axis=0)
        sum_df_1 = ((df_1 - mean_1) ** 2).sum(axis=0)

        fscore = (mean_0 - mean_all) ** 2 + (mean_1 - mean_all) ** 2
        fscore /= 1 / (counts[0] - 1) * sum_df_0 + 1 / (counts[1] - 1) * sum_df_1

        return fscore

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Measure the F-Score for each variable for each dataset.

        :param df_real: the real dataset
        :param df_synthetic: the synthetic dataset
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference **diff_f_score** between averaged real and synthetic F-scores
              across all continuous variables
            * **detailed** -- a dictionary containing the F-scores for the real **real_fscores**
              and synthetic **synthetic_fscores** datasets
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        var_pred = metadata["variable_to_predict"]
        if var_pred is None or len(metadata["continuous"]) == 0:
            return {}

        assert (
            df_real[var_pred].nunique() == 2
        ), "The variable to predict must be binary"
        assert set(df_real[var_pred].unique()) == set(
            df_synthetic[var_pred].unique()
        ), "Datasets must have the same classes"

        # Convert the predicted variable to binary 0/1
        classes = df_real[var_pred].unique()
        classes.sort()
        df_real_trans = df_real.replace({var_pred: {classes[0]: 0, classes[1]: 1}})
        df_synth_trans = df_synthetic.replace(
            {var_pred: {classes[0]: 0, classes[1]: 1}}
        )

        # Compute the fscores for each dataset
        vars = [var_pred] + metadata["continuous"]
        real_fscores = self.fscore(df_real_trans[vars], predicted_var=var_pred)
        synth_fscores = self.fscore(df_synth_trans[vars], predicted_var=var_pred)

        diff = abs(real_fscores - synth_fscores)

        res = {
            "average": {"diff_f_score": np.mean(diff)},
            "detailed": {
                "real_fscores": real_fscores,
                "synthetic_fscores": synth_fscores,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot of the F-scores of both real and synthetic data.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert all(key in report for key in ["real_fscores", "synthetic_fscores"])

        plt.figure(figsize=figsize, layout="constrained")

        udraw.bar_plot_hue(
            s=pd.Series(report["real_fscores"]),
            s_nested=pd.Series(report["synthetic_fscores"]),
            original_name="Real",
            nested_name="Synthetic",
            hue_name="Data",
            title=f"Metric: {cls.name}",
            value_name="F-scores",
            orient="h",
        )
