# Standard library
from typing import List, Tuple, Type
from abc import ABCMeta, abstractmethod
import warnings

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelBinarizer,
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Local
import clover.utils.draw as udraw
import clover.utils.learning as ulearning
from ..base import Metric


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [Regression, Classification, FScore, FeatureImportance]


class Prediction(Metric, metaclass=ABCMeta):
    """
    Check that the synthetic data have the same behavior as the real data regarding the application task.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Prediction"
    alias = "prediction"
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
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
        use_gpu: bool = False,
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._num_kfolds = num_kfolds
        self._num_optuna_trials = num_optuna_trials
        self._use_gpu = use_gpu

    def _learning(
        self,
        x_reference: pd.DataFrame,
        y_reference: np.ndarray,
        x_comparative: pd.DataFrame,
        y_comparative: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
    ) -> dict:
        """
        Train a classifier and score the predictions for the test sets from the reference and comparative inputs.

        :param x_reference: the reference inputs
        :param y_reference: the reference ground truth
        :param x_comparative: the comparative input
        :param y_comparative: the comparative ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
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
                    # "passthrough",  # catboost has its own way to handly categorical variables
                    OneHotEncoder(  # drop first not used since the first category and the unknown would be the same.
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

        if self.__class__.name == "Regression":
            xgboostPredictor = xgb.XGBRegressor
            loss_function = "reg:squarederror"
            scoring_cv = "neg_root_mean_squared_error"
            refit_function = ulearning.refit_rmse_score
        else:
            xgboostPredictor = xgb.XGBClassifier
            loss_function = (
                "multi:softmax"
                if len(np.unique(y_reference)) > 2
                else "binary:logistic"
            )
            scoring_cv = "roc_auc_ovo" if len(np.unique(y_reference)) > 2 else "roc_auc"
            refit_function = ulearning.refit_auc_score

        pipeline = lambda trial: ulearning.pipeline_prediction(
            trial,
            predictor=xgboostPredictor,
            preprocessing=preprocessing,
            loss_function=loss_function,
            use_gpu=self._use_gpu,
        )
        objective = lambda trial: ulearning.objective_cross_val(
            trial,
            pipeline=pipeline,
            df_train=x_reference,
            y_train=y_reference,
            num_kfolds=self._num_kfolds,
            scoring=scoring_cv,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10, seed=np.random.randint(1000)
            ),
        )
        study.optimize(objective, n_trials=self._num_optuna_trials)

        score, _, best_model = refit_function(
            trial=study.best_trial,
            pipeline=pipeline,
            df_train=x_reference,
            y_train=y_reference,
            df_test=x_comparative,
            y_test=y_comparative,
        )

        res = {
            "score_real_test": score,
            "best_model": best_model,
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
        if var_pred in metadata["categorical"] and any(
            df_real["train"][var_pred].value_counts() < self._num_kfolds
        ):  # ensure that there is enough sample in each class
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
            state = np.random.get_state()
            # Prediction RMSE and AUC score on the test set with kfolds
            real_dict = self._learning(
                x_reference=df_train_real,
                y_reference=y_train_real,
                x_comparative=df_test_real,
                y_comparative=y_test_real,
                continuous_cols=cont_cols,
                categorical_cols=cat_cols,
            )

            np.random.set_state(
                state
            )  # ensure that the results are identical for the same datasets
            synth_dict = self._learning(
                x_reference=df_train_synth,
                y_reference=y_train_synth,
                x_comparative=df_test_real,
                y_comparative=y_test_real,
                continuous_cols=cont_cols,
                categorical_cols=cat_cols,
            )

            scores_real_real.append(real_dict["score_real_test"])
            scores_synth_real.append(synth_dict["score_real_test"])

            if var_pred in metadata["continuous"]:  # minimize RMSE
                if real_dict["score_real_test"] <= np.min(scores_real_real):
                    best_model_real = real_dict["best_model"]
                if synth_dict["score_real_test"] <= np.min(scores_synth_real):
                    best_model_synth = synth_dict["best_model"]
            else:  # maximize ROC AUC
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
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Regression"
    alias = "regression"
    score_name = "Root Mean Squared Error"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "diff_real_synth",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "This evaluates the difference in quality (RMSE) of the predictions of a target variable on the real test set made by a regressor trained separately on the real and synthetic datasets.",
                "interpretation": "A difference close to 0 indicates high ML utility for the synthetic dataset.",
            }
        ]

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
    :cvar score_name: the name of the score
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Classification"
    alias = "classif"
    score_name = "AUC score"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "diff_real_synth",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "This evaluates the difference in quality (AUC) of the predictions of a target variable on the real test set made by a classifier trained separately on the real and synthetic datasets.",
                "interpretation": "A difference close to 0 indicates high ML utility for the synthetic dataset.",
            }
        ]

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


class FScore(Metric):
    """
    Check the similarities of the F-scores for each feature of the real and synthetic datasets.

    The F-score is a feature selection technique to evaluate the discrimination potential of a feature.

    :cvar name: the name of the metric
    :vartype name: str
    :cvar alias: the shortname of the metric
    :vartype alias: str

    :param random_state: for reproducibility purposes
    """

    name = "FScore"
    alias = "fscore"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "diff_f_score",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "The absolute difference between averaged real and synthetic F-scores, computed for each continuous variable, measures the preservation of discriminative capacity for synthetic variables. This metric is only computed if the variable to predict is binary.",
                "interpretation": "The lower the difference in F-score, the better discriminative capacity is preserved for continuous variables. Note that variables are not scaled.",
            }
        ]

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


class FeatureImportance(Metric):
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

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Feature Importance"
    alias = "feature_imp"

    def __init__(
        self,
        random_state: int = None,
        num_repeat: int = 10,
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
        use_gpu: bool = False,
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._num_kfolds = num_kfolds
        self._num_optuna_trials = num_optuna_trials
        self._use_gpu = use_gpu

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics formatted as dictionaries
            (**submetric** name, **min**, **max** and **objective**)
        """
        return [
            {
                "submetric": "diff_permutation_importance",
                "min": 0,
                "max": np.inf,
                "objective": "min",
                "description": "This metric evaluates whether importance of each feature for the prediction task is preserved. The feature importance of each variable for the prediction task is calculated using the permutation technique.  The absolute difference between averaged real and synthetic feature importance is reported.",
                "interpretation": "The lower the difference, the better synthetic data preserves the feature importance for the prediction task.",
            }
        ]

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
        if metadata["variable_to_predict"] is None:
            return {}
        if df_real["train"].shape[1] <= 1:
            return {}

        var_pred = metadata["variable_to_predict"]
        independent_vars = list(set(df_real["train"].columns) - {var_pred})

        if var_pred in metadata["continuous"]:
            prediction_class = Regression
            scoring = "neg_root_mean_squared_error"
        else:
            prediction_class = Classification
            scoring = (
                "roc_auc_ovo"
                if len(np.unique(df_real["train"][var_pred])) > 2
                else "roc_auc"
            )

        pred = prediction_class(
            num_repeat=self._num_repeat,
            num_kfolds=self._num_kfolds,
            num_optuna_trials=self._num_optuna_trials,
            use_gpu=self._use_gpu,
        )
        res = pred.compute(df_real, df_synthetic, metadata)
        if len(res) == 0:
            return {}
        res = res["detailed"]

        compute_permutation_importance = lambda dataset: permutation_importance(
            estimator=res[f"best_model_{dataset}"],
            X=res[f"x_test_best_model"],
            y=res[f"y_test_best_model"],
            scoring=scoring,
            n_repeats=5,
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
