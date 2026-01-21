# Standard library
from typing import Union, List, Tuple, Type
from abc import ABCMeta, abstractmethod
from copy import deepcopy

# 3rd party packages
import pandas as pd
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt

# Local
from ..base import Metric
from . import application as app
import clover.utils.learning as ulearning
import clover.utils.draw as udraw


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of utility metrics
    """

    return [Distinguishability, CrossRegression, CrossClassification]


class Distinguishability(Metric):
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

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Distinguishability"
    alias = "dist"

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
                "submetric": "propensity_mse",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "The propensity MSE evaluates the confidence of a classifier (by default, XGBoost) trained to differentiate between real and synthetic data points. The predictions are made on the entire dataset, including training data.",
                "interpretation": "A value of 0 indicates that the model often assigns to the data points a probability of 0.5 of being real: it cannot distinguish confidently between real and synthetic data. Conversely, a value of 1 indicates confidence in the model’s predictions.",
            },
            {
                "submetric": "prediction_mse",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "The prediction MSE evaluates the confidence of the trained classifier when predicting correctly whether the data points are real or synthetic. The predictions are made on the test set only.",
                "interpretation": "A value close to 0 indicates that the model does not predict confidently and correctly whether the data points are real or synthetic. A value of 1 indicates that the model always predicts correctly with full confidence.",
            },
            {
                "submetric": "prediction_auc_rescaled",
                "min": 0,
                "max": 1,
                "objective": "min",
                "description": "The prediction AUC evaluates the performance of the classifier trained to differentiate between real and synthetic data points. The predictions are made on the test set only. The AUC is rescaled to fall in the 0-1 range.",
                "interpretation": "A value of 0 indicates predictions equivalent to random guessing. A score of 1 indicates complete distinguishability.",
            },
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
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        optimize_xgb: bool = True,
    ) -> dict:
        """
        Compute three distinguishability metrics between real and synthetic datasets:
        the propensity mean squared error, the prediction mean squared error and
        the prediction auc score.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :param optimize_xgb: whether the XGB Predictor should be optimized with Optuna
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
        if df_real["train"].shape[1] <= 1:
            return {}

        # Transform original dataframes into input and output arrays for the training stage
        df_train = pd.concat(
            [df_real["train"], df_synthetic["train"]], axis=0, ignore_index=True
        )
        df_test = pd.concat(
            [df_real["test"], df_synthetic["test"]], axis=0, ignore_index=True
        )

        #   Transform the categorical variables to object
        df_train[metadata["categorical"]] = df_train[metadata["categorical"]].astype(
            "object"
        )
        df_test[metadata["categorical"]] = df_test[metadata["categorical"]].astype(
            "object"
        )
        cat_features = list(
            range(  # the columns follow the order specified in the ColumnTransformer
                len(metadata["continuous"]), df_train.shape[1]
            )
        )

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), metadata["continuous"]),
                (
                    "categorical",
                    # "passthrough",  # catboost has its own way to handly categorical variables
                    OneHotEncoder(  # drop first not used since the first category and the unknown would be the same.
                        categories=[
                            df_train[cat].unique() for cat in metadata["categorical"]
                        ],
                        handle_unknown="ignore",
                    ),
                    metadata["categorical"],
                ),
            ],
            verbose_feature_names_out=False,
        )

        #   Label 1 for real records and 0 for synthetic ones
        y_train = np.array(
            [1] * len(df_real["train"]) + [0] * len(df_synthetic["train"])
        )
        y_test = np.array([1] * len(df_real["test"]) + [0] * len(df_synthetic["test"]))

        #   Shuffle train dataset
        df_train["y"] = y_train
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        y_train = df_train["y"].to_numpy()
        df_train = df_train.drop(columns="y")

        # Compute the distinguishability score in three different settings
        dist_scores = []
        prediction_real = []
        prediction_synth = []

        pipeline = lambda trial: ulearning.pipeline_prediction(
            trial,
            predictor=xgb.XGBClassifier,
            preprocessing=preprocessing,
            loss_function="binary:logistic",
            use_gpu=self._use_gpu,
        )
        objective = lambda trial: ulearning.objective_cross_val(
            trial,
            pipeline=pipeline,
            df_train=df_train,
            y_train=y_train,
            num_kfolds=self._num_kfolds,
            scoring="roc_auc",
        )

        # Compute scores several times to account for randomness
        for _ in range(self._num_repeat):
            if optimize_xgb:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=10, seed=np.random.randint(1000)
                    ),
                )
                study.optimize(objective, n_trials=self._num_optuna_trials)
                best_trial = study.best_trial
            else:
                # Use default XGBClassifier without optimization
                best_trial = optuna.trial.FixedTrial(
                    {
                        "eta": 0.1,
                        "max_depth": 6,
                        "subsample": 0.5,
                        "colsample_bylevel": 1,
                    }
                )

            #   Shuffle train dataset - TO DELETE
            df_train["y"] = y_train
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            y_train = df_train["y"].to_numpy()
            df_train = df_train.drop(columns="y")

            auc, y_pred_proba, _ = ulearning.refit_auc_score(
                trial=best_trial,
                pipeline=pipeline,
                df_train=df_train,
                y_train=y_train,
                df_test=df_test,
                y_test=y_test,
            )

            mse = self.propensity_mse(y_pred_proba)
            y_pred_real = y_pred_proba[: len(df_test) // 2]
            mse_real = self.propensity_mse(y_pred_real)
            y_pred_synth = y_pred_proba[len(df_test) // 2 :]
            mse_synth = self.propensity_mse(y_pred_synth)
            auc_rescaled = max(0.5, auc) * 2 - 1  # scale between 0 and 1
            pred_mse = (
                1
                / 2
                * (
                    self.propensity_mse(np.maximum(0.5, y_pred_real))
                    + self.propensity_mse(np.minimum(0.5, y_pred_synth))
                )
            )

            # Average scores on kfolds
            dist_scores.append([mse, mse_real, mse_synth, auc_rescaled, pred_mse])
            prediction_real.extend(y_pred_real)
            prediction_synth.extend(y_pred_synth)

        dist_scores = np.array(dist_scores)

        # Average scores on repetitions
        (
            propensity_mse,
            prediction_mse_real,
            prediction_mse_synth,
            prediction_auc_rescaled,
            prediction_mse,
        ) = np.mean(dist_scores, axis=0)

        res = {
            "average": {
                "propensity_mse": propensity_mse,
                "prediction_mse_real": prediction_mse_real,
                "prediction_mse_synth": prediction_mse_synth,
                "prediction_auc_rescaled": prediction_auc_rescaled,
                "prediction_mse": prediction_mse,
            },
            "detailed": {
                "propensity_mse": dist_scores[:, 0],
                "prediction_mse_real": dist_scores[:, 1],
                "prediction_mse_synth": dist_scores[:, 2],
                "prediction_auc_rescaled": dist_scores[:, 3],
                "prediction_mse": dist_scores[:, 4],
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
                "prediction_auc_rescaled",
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
                "prediction_auc_rescaled": report["prediction_auc_rescaled"],
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


class CrossLearning(Metric, metaclass=ABCMeta):
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
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Cross"
    alias = "cross"
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
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
        use_gpu: bool = False,
    ):
        super().__init__(random_state)
        self._num_repeat = num_repeat
        self._num_kfolds = num_kfolds
        self._num_optuna_trials = num_optuna_trials
        self._use_gpu = use_gpu

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Compare the real and synthetic test sets predictions
        when the model is trained on the real dataset or the synthetic one.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train** and **test** sets
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
        if df_real["train"].shape[1] <= 1:
            return {}

        dic_realtrain_realtest = {}
        dic_synthtrain_realtest = {}
        diff_real_synth = []

        for col in dependent_vars:
            metadata_pred = deepcopy(metadata)
            metadata_pred["variable_to_predict"] = col

            pred = getattr(app, self.__class__.class_name)(
                num_repeat=self._num_repeat,
                num_kfolds=self._num_kfolds,
                num_optuna_trials=self._num_optuna_trials,
                use_gpu=self._use_gpu,
            )
            res = pred.compute(df_real, df_synthetic, metadata_pred)
            if len(res) == 0:
                continue
            else:
                res = res["detailed"]

            dic_realtrain_realtest[col] = res["score_real_real"]
            dic_synthtrain_realtest[col] = res["score_synth_real"]

            # Absolute difference for the average report
            diff_real_synth.append(
                abs(dic_realtrain_realtest[col] - dic_synthtrain_realtest[col])
            )

        if len(dic_realtrain_realtest) == 0:
            return {}

        res = {
            "average": {
                "diff_real_synth": np.mean(diff_real_synth),
            },
            "detailed": {
                "real_real": dic_realtrain_realtest,
                "synth_real": dic_synthtrain_realtest,
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
                "synth_real",
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
                columns=[
                    "Trained real test real",
                    "Trained synthetic test real",
                ],
            )

            udraw.bar_plot(
                data=df,
                orient="v",
                title="",
                value_name=getattr(app, cls.class_name).score_name,
                xrotation=True,
                ax=axes[i],
            )
            axes[i].set_title(col)


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
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    """

    name = CrossLearning.name + " Regression"
    alias = CrossLearning.alias + "_reg"
    class_name = "Regression"

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
            }
        ]


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
    :cvar class_name: the prediction class name
    :vartype score_name: str

    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    """

    name = CrossLearning.name + " Classification"
    alias = CrossLearning.alias + "_classif"
    class_name = "Classification"

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
            }
        ]
