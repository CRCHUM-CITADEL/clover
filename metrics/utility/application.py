#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import List, Tuple, Type  # standard library
from abc import ABCMeta, abstractmethod
import warnings

import pandas as pd  # 3rd party packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

import utils.draw as udraw  # local
import utils.learning as ulearning
from metrics.utility.base import UtilityMetric


def get_metrics() -> List[Type[UtilityMetric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [Regression, Classification, FScore, FeatureImportance]


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
        x_reference: pd.DataFrame,
        y_reference: np.ndarray,
        x_comparative: pd.DataFrame,
        y_comparative: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        target: pd.Series,
        num_folds: int,
    ) -> dict:
        """
        Train a classifier or a regressor and score the predictions for the test sets
        from the reference and comparative inputs. To be reimplemented.

        :param x_reference: the reference inputs
        :param y_reference: the reference ground truth
        :param x_comparative: the comparative input
        :param y_comparative: the comparative ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: a dictionary containing the average scores **score_reference** and **score_comparative** on k-folds,
          the **best_model** and the testing sets **x_test_best_model** and **y_test_best_model**
          used for the best model
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
        cat_cols = [
            col
            for col in df_real.columns
            if col not in metadata["continuous"] + [var_pred]
        ]
        cont_cols = [col for col in df_real.columns if col not in cat_cols + [var_pred]]

        # Compute the cross learning in both directions
        scores_real_real = []
        scores_real_synth = []
        scores_synth_synth = []
        scores_synth_real = []
        best_model_real = None
        x_test_best_model_real = None
        y_test_best_model_real = None
        best_model_synth = None
        x_test_best_model_synth = None
        y_test_best_model_synth = None

        learning = lambda reference_tuple, comparative_tuple: self._learning(
            x_reference=reference_tuple[0],
            y_reference=reference_tuple[1],
            x_comparative=comparative_tuple[0],
            y_comparative=comparative_tuple[1],
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
            target=df_real[var_pred],
            num_folds=self._num_folds,
        )

        # Compute scores several times to account for randomness
        for i in range(self._num_repeat):
            # Prediction MSE and AUC score on the test set with kfolds
            state = np.random.get_state()
            real_dict = learning(
                reference_tuple=(df_real_x, y_real),
                comparative_tuple=(df_synthetic_x, y_synth),
            )

            np.random.set_state(state)  # ensure same beginning
            synth_dict = learning(
                reference_tuple=(df_synthetic_x, y_synth),
                comparative_tuple=(df_real_x, y_real),
            )

            #  TODO: remove repetitions and use ref/comp
            scores_real_real.append(real_dict["score_reference"])
            scores_real_synth.append(real_dict["score_comparative"])
            scores_synth_synth.append(synth_dict["score_reference"])
            scores_synth_real.append(synth_dict["score_comparative"])

            if real_dict["score_reference"] >= np.max(scores_real_real):
                best_model_real = real_dict["best_model"]
                x_test_best_model_real = real_dict["x_test_best_model"]
                y_test_best_model_real = real_dict["y_test_best_model"]
            if synth_dict["score_reference"] >= np.max(scores_synth_synth):
                best_model_synth = synth_dict["best_model"]
                x_test_best_model_synth = synth_dict["x_test_best_model"]
                y_test_best_model_synth = synth_dict["y_test_best_model"]

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
                "best_model_real": best_model_real,
                "best_model_synth": best_model_synth,
                "x_test_best_model_real": x_test_best_model_real,
                "y_test_best_model_real": y_test_best_model_real,
                "x_test_best_model_synth": x_test_best_model_synth,
                "y_test_best_model_synth": y_test_best_model_synth,
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
        x_reference: pd.DataFrame,
        y_reference: np.ndarray,
        x_comparative: pd.DataFrame,
        y_comparative: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        target: pd.Series,
        num_folds: int,
    ) -> dict:
        """
        Train a regressor and score the predictions for the test sets from the reference and comparative inputs.

        :param x_reference: the reference inputs
        :param y_reference: the reference ground truth
        :param x_comparative: the comparative input
        :param y_comparative: the comparative ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: a dictionary containing the average scores **score_reference** and **score_comparative** on k-folds,
          the **best_model** and the testing sets **x_test_best_model** and **y_test_best_model**
          used for the best model
        """

        mse_ref = []
        mse_comp = []
        best_pipe = None
        x_test_best_model = None
        y_test_best_model = None

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), continuous_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        drop="first",
                        categories=[
                            list(
                                set(x_reference[cat].unique())
                                | set(x_comparative[cat].unique())
                            )
                            for cat in categorical_cols
                        ],
                    ),  # TODO: use infrequent categories?
                    categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

        kf = KFold(n_splits=num_folds, shuffle=True)
        for train_index, test_index in kf.split(x_reference, y_reference):
            pipe = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("gbm", GradientBoostingRegressor()),
                ]
            )
            scores, _ = ulearning.train_predict(
                pipeline=pipe,
                x_train=x_reference.iloc[train_index],
                y_train=y_reference[train_index],
                x_test_list=[
                    x_reference.iloc[test_index],
                    x_comparative.iloc[test_index],
                ],
                y_test_list=[y_reference[test_index], y_comparative[test_index]],
            )
            mse_ref.append(scores[0])
            mse_comp.append(scores[1])

            if scores[0] >= np.max(mse_ref):
                best_pipe = pipe
                x_test_best_model = x_reference.iloc[test_index]
                y_test_best_model = y_reference[test_index]

        res = {
            "score_reference": np.mean(mse_ref),
            "score_comparative": np.mean(mse_comp),
            "best_model": best_pipe,
            "x_test_best_model": x_test_best_model,
            "y_test_best_model": y_test_best_model,
        }

        return res

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
        x_reference: pd.DataFrame,
        y_reference: np.ndarray,
        x_comparative: pd.DataFrame,
        y_comparative: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        target: pd.Series,
        num_folds: int,
    ) -> dict:
        """
        Train a classifier and score the predictions for the test sets from the reference and comparative inputs.

        :param x_reference: the reference inputs
        :param y_reference: the reference ground truth
        :param x_comparative: the comparative input
        :param y_comparative: the comparative ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
        :param target: the target to extract labels if a classifier is trained
        :param num_folds: the scores are averaged across the number of folds to account for split randomness
        :return: a dictionary containing the average scores **score_reference** and **score_comparative** on k-folds,
          the **best_model** and the testing sets **x_test_best_model** and **y_test_best_model**
          used for the best model
        """

        labels = target.unique()
        labels.sort()

        auc_ref = []
        auc_comp = []
        best_pipe = None
        x_test_best_model = None
        y_test_best_model = None

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), continuous_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        drop="first",
                        categories=[
                            list(
                                set(x_reference[cat].unique())
                                | set(x_comparative[cat].unique())
                            )
                            for cat in categorical_cols
                        ],
                    ),  # TODO: use infrequent categories?
                    categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        for train_index, test_index in kf.split(x_reference, y_reference):
            pipe = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("gbm", GradientBoostingClassifier()),
                ]
            )
            scores, _ = ulearning.train_predict(
                pipeline=pipe,
                x_train=x_reference.iloc[train_index],
                y_train=y_reference[train_index],
                x_test_list=[
                    x_reference.iloc[test_index],
                    x_comparative.iloc[test_index],
                ],
                y_test_list=[y_reference[test_index], y_comparative[test_index]],
                classif_labels=labels,
            )
            auc_ref.append(scores[0])
            auc_comp.append(scores[1])

            if scores[0] >= np.max(auc_ref):
                best_pipe = pipe
                x_test_best_model = x_reference.iloc[test_index]
                y_test_best_model = y_reference[test_index]

        res = {
            "score_reference": np.mean(auc_ref),
            "score_comparative": np.mean(auc_comp),
            "best_model": best_pipe,
            "x_test_best_model": x_test_best_model,
            "y_test_best_model": y_test_best_model,
        }

        return res

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


class FeatureImportance(UtilityMetric):
    """
    Check the importance of each feature for the prediction task is preserved.

    Based on the Permutation Importance technique. The values of each feature are shuffled
    and the impact on the prediction is measured and used as the feature importance score.
    This method is agnostic to the model.

    .. warning:: Correlations affect the importance score and should be considered when reading the results.

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
    """

    name = "Feature Importance"
    alias = "feature_imp"
    min = 0
    max = np.inf
    objective = "min"

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
        return ["diff_permutation_importance"]

    def compute(
        self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
    ) -> dict:
        """
        Measure the Permutation Importance score for each variable for each dataset.

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
        if df_real.shape[1] <= 1:
            return {}

        var_pred = metadata["variable_to_predict"]
        independent_vars = list(set(df_real.columns) - {var_pred})

        prediction_class = (
            Regression if var_pred in metadata["continuous"] else Classification
        )

        pred = prediction_class(num_repeat=self._num_repeat, num_folds=self._num_folds)
        res = pred.compute(df_real, df_synthetic, metadata)
        if len(res) == 0:
            return {}
        res = res["detailed"]

        compute_permutation_importance = lambda dataset: permutation_importance(
            estimator=res[f"best_model_{dataset}"],
            X=res[f"x_test_best_model_{dataset}"],
            y=res[f"y_test_best_model_{dataset}"],
            scoring=None,  # the one used by the estimator
            n_repeats=20,
        )

        real_importance = compute_permutation_importance(dataset="real")
        synth_importance = compute_permutation_importance(dataset="synth")

        real_importance_mean = real_importance.importances_mean
        synth_importance_mean = synth_importance.importances_mean

        diff = abs(real_importance_mean - synth_importance_mean)
        real_importance_mean_series = pd.Series(
            real_importance_mean, index=independent_vars
        )
        synth_importance_mean_series = pd.Series(
            synth_importance_mean, index=independent_vars
        )

        res = {
            "average": {"diff_permutation_importance": np.mean(diff)},
            "detailed": {
                "real_permutation_importance": real_importance_mean_series,
                "synthetic_permutation_importance": synth_importance_mean_series,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot of the permutation importances of both real and synthetic data.

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """

        assert report is not None
        assert all(
            key in report
            for key in [
                "real_permutation_importance",
                "synthetic_permutation_importance",
            ]
        )

        plt.figure(figsize=figsize, layout="constrained")

        udraw.bar_plot_hue(
            s=report["real_permutation_importance"],
            s_nested=report["synthetic_permutation_importance"],
            original_name="Real",
            nested_name="Synthetic",
            hue_name="Data",
            title=f"Metric: {cls.name}",
            value_name="Permutation importance",
            orient="h",
        )
