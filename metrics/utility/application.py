# Standard library
from typing import List, Tuple, Type
from abc import ABCMeta, abstractmethod
import warnings

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelBinarizer,
    StandardScaler,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb

# Local
import utils.draw as udraw
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
    :param use_gpu: flag to use GPU computation power to accelerate the learning
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
        use_gpu: bool = False,
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._use_gpu = use_gpu

    @classmethod
    def get_average_submetrics(cls) -> List[str]:
        """
        Get the average submetrics of the current metric.

        :return: the list of the average submetrics
        """
        return ["diff_real_synth"]

    @classmethod
    def _learning(
        cls,
        x_reference: pd.DataFrame,
        y_reference: np.ndarray,
        x_comparative: pd.DataFrame,
        y_comparative: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        use_gpu: bool,
    ) -> dict:
        """
        Train a classifier and score the predictions for the test sets from the reference and comparative inputs.

        :param x_reference: the reference inputs
        :param y_reference: the reference ground truth
        :param x_comparative: the comparative input
        :param y_comparative: the comparative ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
        :param use_gpu: flag to use GPU computation power to accelerate the learning
        :return: a dictionary containing the average scores **score_reference** and **score_comparative** on k-folds,
          the **best_model** and the testing sets **x_test_best_model** and **y_test_best_model**
          used for the best model
        """

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), continuous_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        # drop="first", not used since the first category and the unknown would be the same.
                        categories=[
                            x_reference[cat].unique() for cat in categorical_cols
                        ],
                        handle_unknown="ignore",
                    ),
                    categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

        if cls.name == "Regression":
            xgbPredictor = xgb.XGBRegressor
            objective = "reg:squarederror"
        else:
            xgbPredictor = xgb.XGBClassifier
            objective = (
                "multi:softprob"
                if len(np.unique(y_reference)) > 2
                else "binary:logistic"
            )

        model_params = {
            "n_estimators": 100,
            "eta": 0.1,
            "tree_method": "auto" if not use_gpu else "gpu_hist",
            "objective": objective,
        }

        pipe = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("xgb", xgbPredictor(**model_params)),
            ]
        )
        scores, _ = ulearning.train_predict(
            pipeline=pipe,
            x_train=x_reference,
            y_train=y_reference,
            x_test_list=[x_comparative],
            y_test_list=[y_comparative],
            is_classification=cls.name == "Classification",
        )

        res = {
            "score_real_test": scores[0],
            "best_model": pipe,
        }

        return res

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
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
        if df_real["train"].shape[1] <= 1:
            return {}

        # Create x and y data
        df_train_real = df_real["train"].drop(columns=var_pred)
        df_train_synth = df_synthetic["train"].drop(columns=var_pred)
        df_test_real = df_real["test"].drop(columns=var_pred)
        y_train_real = df_real["train"][var_pred].to_numpy()
        y_train_synth = df_synthetic["train"][var_pred].to_numpy()
        y_test_real = df_real["test"][var_pred].to_numpy()

        # Transform categorical columns in one-hot format
        #   Ensure the output is binary if there are two classes
        if var_pred in metadata["categorical"]:
            if len(np.unique(y_train_real)) == 2:
                lenc = LabelBinarizer()
            else:
                lenc = LabelEncoder()
            lenc.fit(y_train_real)
            y_train_real = lenc.transform(y_train_real).flatten()
            y_train_synth = lenc.transform(y_train_synth).flatten()
            y_test_real = lenc.transform(y_test_real).flatten()

        #   Select the categorical columns to transform
        cat_cols = [
            col
            for col in df_real["train"].columns
            if col not in metadata["continuous"] + [var_pred]
        ]
        cont_cols = [
            col for col in df_real["train"].columns if col not in cat_cols + [var_pred]
        ]
        df_train_real[cat_cols] = df_train_real[cat_cols].astype("object")
        df_train_synth[cat_cols] = df_train_synth[cat_cols].astype("object")
        df_test_real[cat_cols] = df_test_real[cat_cols].astype("object")

        # Compute the cross learning in both directions
        scores_real_real = []
        scores_synth_real = []
        best_model_real = None
        best_model_synth = None

        # Compute scores several times to account for randomness
        for i in range(self._num_repeat):
            # Prediction MSE and AUC score on the test set with kfolds
            real_dict = self._learning(
                x_reference=df_train_real,
                y_reference=y_train_real,
                x_comparative=df_test_real,
                y_comparative=y_test_real,
                continuous_cols=cont_cols,
                categorical_cols=cat_cols,
                use_gpu=self._use_gpu,
            )

            synth_dict = self._learning(
                x_reference=df_train_synth,
                y_reference=y_train_synth,
                x_comparative=df_test_real,
                y_comparative=y_test_real,
                continuous_cols=cont_cols,
                categorical_cols=cat_cols,
                use_gpu=self._use_gpu,
            )

            scores_real_real.append(real_dict["score_real_test"])
            scores_synth_real.append(synth_dict["score_real_test"])

            if real_dict["score_real_test"] >= np.max(scores_real_real):
                best_model_real = real_dict["best_model"]
            if synth_dict["score_real_test"] >= np.max(scores_synth_real):
                best_model_synth = synth_dict["best_model"]

        diff_real_synth = abs(np.array(scores_real_real) - np.array(scores_synth_real))

        res = {
            "average": {
                "diff_real_synth": np.mean(diff_real_synth),
            },
            "detailed": {
                "score_real_real": np.array(scores_real_real),
                "score_synth_real": np.array(scores_synth_real),
                "best_model_real": best_model_real,
                "best_model_synth": best_model_synth,
                "x_test_best_model": df_test_real,
                "y_test_best_model": y_test_real,
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
        assert all(key in report for key in ["score_real_real", "score_synth_real"])

        plt.figure(figsize=figsize, layout="constrained")

        df = pd.DataFrame(
            np.column_stack(
                (
                    report["score_real_real"],
                    report["score_synth_real"],
                )
            ),
            columns=[
                "Trained real test real",
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
    XGBRegressor is used for the learning task.

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
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Regression"
    alias = "regression"
    score_name = "Mean Squared Error"

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
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
    XGBClassifier is used for the learning task.

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
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Classification"
    alias = "classif"
    max = 1
    score_name = "AUC score"

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset and the synthetic one respectively.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference between the real and the synthetic performances
            * **detailed** -- a dictionary containing the scores for real and synthetic test datasets
        """

        var_pred = metadata["variable_to_predict"]
        if var_pred is None or var_pred in metadata["continuous"]:
            return {}

        if set(df_real["train"][var_pred].unique()) != set(
            df_synthetic["train"][var_pred].unique()
        ):
            warnings.warn(
                message=f"The datasets do not have the same labels for the variable {var_pred}. "
                f"The metric {self.name} cannot be computed.",
                category=UserWarning,
            )
            return {}

        if (
            df_real["train"][var_pred].nunique() == 1
            or df_synthetic["train"][var_pred].nunique() == 1
        ):
            warnings.warn(
                message=f"There is only one class in the variable {var_pred}. "
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
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Measure the F-Score for each variable for each dataset.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
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
        if var_pred not in metadata["categorical"]:
            return {}
        if df_real["test"][var_pred].nunique() != 2:
            return {}

        assert set(df_real["test"][var_pred].unique()) == set(
            df_synthetic["test"][var_pred].unique()
        ), "Datasets must have the same classes"

        # Convert the predicted variable to binary 0/1
        classes = df_real["test"][var_pred].unique()
        classes.sort()
        df_real_trans = df_real["test"].replace(
            {var_pred: {classes[0]: 0, classes[1]: 1}}
        )
        df_synth_trans = df_synthetic["test"].replace(
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
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat

    @classmethod
    def get_average_submetrics(cls) -> List[str]:
        """
        Get the average submetrics of the current metric.

        :return: the list of the average submetrics
        """
        return ["diff_permutation_importance"]

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Measure the Permutation Importance score for each variable for each dataset.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the absolute difference **diff_f_score** between averaged real and synthetic F-scores
              across all continuous variables
            * **detailed** -- a dictionary containing the F-scores for the real **real_fscores**
              and synthetic **synthetic_fscores** datasets
        """

        super().check_consistency_compute_parameters(df_real, df_synthetic, metadata)
        if df_real["train"].shape[1] <= 1:
            return {}

        var_pred = metadata["variable_to_predict"]
        independent_vars = list(set(df_real["train"].columns) - {var_pred})

        prediction_class = (
            Regression if var_pred in metadata["continuous"] else Classification
        )

        pred = prediction_class(num_repeat=self._num_repeat)
        res = pred.compute(df_real, df_synthetic, metadata)
        if len(res) == 0:
            return {}
        res = res["detailed"]

        compute_permutation_importance = lambda dataset: permutation_importance(
            estimator=res[f"best_model_{dataset}"],
            X=res[f"x_test_best_model"],
            y=res[f"y_test_best_model"],
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
