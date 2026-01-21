# Standard library
from typing import Tuple, List, Type, Any
from abc import ABCMeta
import warnings

# 3rd party packages
import pandas as pd
import numpy as np
from numpy import ndarray, dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local
from ..base import Metric
import clover.utils.draw as udraw
import clover.utils.external.gower.gower_dist as gower
import clover.utils.learning as ulearning


def get_metrics() -> List[Type[Metric]]:
    """
    List all the available metrics in this module.

    :return: a list of the classes of privacy metrics (membership inference attacks)
    """

    return [GANLeaks, MCMembership, Logan, TableGan, Detector, Collision]


class AttackModel(Metric, metaclass=ABCMeta):
    """
    Membership inference attacks.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real dataset
        for better computing performance when evaluating the model (detector and collision are not affected by this parameter)
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness (ganleaks and MC memebership are not affected by this parameter)
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier (ganleaks and MC memebership are not affected by this parameter)
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters (ganleaks and MC memebership are not affected by this parameter)
    :param use_gpu: flag to use GPU computation power to accelerate the learning (ganleaks and MC memebership are not affected by this parameter)
    """

    name = "AttackModel"
    alias = "attack"

    def __init__(
        self,
        random_state: int = None,
        sampling_frac: float = 0.5,
        num_repeat: int = 10,
        num_kfolds: int = 5,
        num_optuna_trials: int = 20,
        use_gpu: bool = False,
    ):
        super().__init__(random_state)
        self._random_state = random_state
        self._sampling_frac = sampling_frac
        self._num_repeat = num_repeat
        self._num_kfolds = num_kfolds
        self._num_optuna_trials = num_optuna_trials
        self._use_gpu = use_gpu

    @staticmethod
    def check_consistency_compute_parameters(
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
        train_test_ref: bool = False,
    ) -> None:
        """
        Assert that the compute method parameters are consistent, when attack model is applied.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :param train_test_ref: a boolean parameter indicating whether the metric is calculated for synthetic data
          or for the test set as a reference.
        :return: *None*
        """

        Metric.check_consistency_compute_parameters(
            df_real, df_synthetic, metadata, train_test_ref
        )

        assert (
            df_synthetic["train"].shape == df_synthetic["2nd_gen"].shape
        ), "1st generation synthetic train set and 2nd generation synthetic set must have the same shape"

        assert set(df_synthetic["train"].columns) == set(
            df_synthetic["2nd_gen"].columns
        ), "1st generation synthetic train set and 2nd generation synthetic set must have the same columns"

        assert (
            df_synthetic["train"].shape[0] > df_synthetic["test"].shape[0]
        ), "In order to train TableGAN, there should be more samples in 1st generation synthetic train set than 1st generation synthetic test set"

        assert (df_real["test"] is not None) and (
            df_real["test"].shape[0] > 0
        ), "Control set should not be empty"

    @classmethod
    def precision_top_n(
        cls, n: int, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        """Compute the top n% precision score

        :param n: top n% of the prediction used to compute the precision score
        :param y_true: the ground truth
        :param y_pred_proba: the predicted probability of the positive class
        :return: top n% precision score
        """

        y_pred_proba_decend_idx = y_pred_proba.argsort()[::-1]
        y_true_top_n = y_true[y_pred_proba_decend_idx][
            : int(len(y_pred_proba_decend_idx) * (n / 100))
        ]

        if len(y_true_top_n) > 0:
            return np.count_nonzero(y_true_top_n) / len(y_true_top_n)
        else:
            warnings.warn(
                f"Not enough samples in test set to compute top {n}% precision"
            )
            return np.nan

    @classmethod
    def tpr_at_n_fpr(cls, n: float, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Compute true positive rate at n% false positive rate

        :param n: the false positive rate at which the true positive rate is extracted
        :param fpr: increasing false positive rates
        :param tpr: corresponding increasing true positive rates
        :return: tps at n% fpr
        """
        target_tpr = tpr[np.where(fpr < n / 100)[0][-1]]

        return target_tpr

    @classmethod
    def compute_frequency(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the frequency of each record in a dataframe and add it as a column

        :param df: the original dataframe
        :return: new dataframe with frequency
        """

        df_copy = (
            df.copy()
        )  # Make a copy so that the original dataframe is not modified
        df_copy["frequency"] = df_copy.groupby(
            list(df_copy.columns), observed=True
        ).transform("size") / len(df_copy)

        df_copy.drop_duplicates(inplace=True, ignore_index=True)

        return df_copy

    @classmethod
    def annotate_collision(
        cls, df_target: pd.DataFrame, df_ref: pd.DataFrame
    ) -> pd.DataFrame:
        """Add a column to the target dataframe to indicate if each record exists in the reference dataframe.
        Note: the original dataframe is not modified

        :param df_target: the target dataframe
        :param df_ref: the reference dataframe
        :return: the target dataframe with a new column added to indicate collision
        """

        df_merge = pd.merge(
            df_target,
            df_ref,
            on=df_target.columns.tolist(),
            indicator=True,
            how="left",
        )

        df_merge["collision"] = df_merge.apply(
            lambda row: 1 if row["_merge"] == "both" else 0, axis=1
        )

        df_target_new = pd.merge(
            df_target,
            df_merge.drop_duplicates(),
            on=df_target.columns.tolist(),
            how="left",
        )

        return df_target_new

    def hyperparam_tuning(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
    ) -> Pipeline:
        """
        Train a classifier with hyperparameters tuning.

        :param x: the inputs
        :param y: the ground truth
        :param continuous_cols: the continuous columns
        :param categorical_cols: the categorical columns
        :return: the **best_model** with the optimized hyperparameters
        """

        # ColumnTransformers
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), continuous_cols),
                (
                    "categorical",
                    # "passthrough",  # catboost has its own way to handle categorical variables
                    OneHotEncoder(
                        # drop="first", not used since the first category and the unknown would be the same.
                        categories=[x[cat].unique() for cat in categorical_cols],
                        handle_unknown="ignore",
                    ),
                    categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

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
            df_train=x,
            y_train=y,
            num_kfolds=self._num_kfolds,
            scoring="roc_auc",
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10, seed=np.random.randint(1000)
            ),
        )
        study.optimize(objective, n_trials=self._num_optuna_trials)

        # Refit model with best hyperparameters
        best_pipe = pipeline(study.best_trial)
        best_pipe.fit(x, y)

        return best_pipe


class GANLeaks(AttackModel):
    """
    Membership inference attacks by GAN-Leaks.

    See `Chen, Dingfan, Ning Yu, Yang Zhang, and Mario Fritz.
    "Gan-leaks: A taxonomy of membership inference attacks against generative models."
    In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security,
    pp. 343-362. 2020. <https://dl.acm.org/doi/abs/10.1145/3372297.3417238>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real dataset
        for better computing performance when evaluating the model
    """

    name = "GAN-Leaks"
    alias = "ganleaks"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision_top1%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision_top50%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
        ]
        return submetrics

    def eval(
        self,
        df_test: pd.DataFrame,
        y_test: np.ndarray,
        df_synth: pd.DataFrame,
        cat_cols: list,
    ) -> tuple[
        float, float, ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], Any, Any
    ]:
        """
        Evaluate a GAN-Leaks model

        :param df_test: the real data to be evaluated
        :param y_test: the true label of the real data
        :param df_synth: the reference synthetic data
        :param cat_cols: the columns with categorical variables
        :return: top 1% precision and top 50% precision of the predictions,
            DCRs for the top 1% and top 50% predictions,
            predicted probabilities for each record and
            DCRs for each record
        """
        df_synth_ref = df_synth[df_test.columns]  # gower needs the same order

        cat_features = [
            True if col in cat_cols else False for col in df_test.columns
        ]  # boolean array instead of column names

        # Compute Gower distance (adapted to mixed data): data_y is data to compare
        pairwise_gower = gower.gower_matrix(
            data_x=df_test, data_y=df_synth_ref, cat_features=cat_features
        )

        # Fetch the shortest distance for each record
        min_dist = np.min(pairwise_gower, axis=1)

        # Convert the distance to probability, in order to compute top 1% and top 50% precision
        y_pred_proba = np.exp(-min_dist)

        distance_sorted = np.sort(min_dist)
        distance_top1 = distance_sorted[: int(len(distance_sorted) * 0.01)]
        distance_top50 = distance_sorted[: int(len(distance_sorted) * 0.5)]

        # Compute the metrics
        precision_top1 = self.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba
        )
        precision_top50 = self.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba
        )

        return (
            precision_top1,
            precision_top50,
            distance_top1,
            distance_top50,
            min_dist,
            y_pred_proba,
        )

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Membership inference attacks based on the distance to the closest record. Evaluate with real data,
        which consists of the real data used to generate 1st generation synthetic data and a control (test set).

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision** and **top 50 precision**
                to predict if a record in the real set (train and test) is used to generate the first generation synthetic data
            * **detailed** -- distance to the closest record for each test sample to the closest synthetic sample
                for top 1% and top 50% prediction **top 1% distance** and **top 50% distance**
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Sample a fraction of the datasets for computation performance.
        # The size of real_train and real_control should be the same, so that the test set is balanced.
        real_control = df_real["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        real_train = df_real["train"].sample(
            n=len(real_control),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        df_test = pd.concat([real_train, real_control], axis=0, ignore_index=True)

        print(f"GAN-Leaks test set shape: {df_test.shape}")

        # Label 1 for real records used to generate 1st generation synthetic data and 0 for control
        y_test = np.array([1] * len(real_train) + [0] * len(real_control))

        (
            precision_top1,
            precision_top50,
            distance_top1,
            distance_top50,
            y_pred_proba,
            _,
        ) = self.eval(
            df_test=df_test,
            y_test=y_test,
            df_synth=df_synthetic["train"],
            cat_cols=metadata["categorical"],
        )

        res = {
            "average": {
                "precision_top1%": precision_top1,
                "precision_top50%": precision_top50,
            },
            "detailed": {
                "distance_top1%": distance_top1,
                "distance_top50%": distance_top50,
                "prediction": {
                    "test_index": df_test.index,
                    "y_test": y_test,
                    "model_pred": y_pred_proba,
                },
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a histogram for distance to closest record for top 1% and top 50% prediction

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "distance_top1%",
                "distance_top50%",
            ]
        )

        bounds_values = np.concatenate(
            (
                report["distance_top1%"],
                report["distance_top50%"],
            )
        )

        mini = np.min(bounds_values)
        maxi = np.max(bounds_values)

        _, axes = plt.subplots(
            ncols=2,
            nrows=1,
            figsize=figsize,
            layout="constrained",
            sharex="row",
            sharey="row",
        )

        plt.suptitle("GAN-Leaks")

        udraw.histplot_plot(
            s=pd.Series(report["distance_top1%"]),
            title="top1%",
            value_name="Gower Distance to Closest Record",
            stat="proportion",
            bins=10,
            binrange=(mini, maxi),
            xrotation=False,
            ax=axes[0],
        )

        udraw.histplot_plot(
            s=pd.Series(report["distance_top50%"]),
            title="top50%",
            value_name="Gower Distance to Closest Record",
            stat="proportion",
            bins=10,
            binrange=(mini, maxi),
            xrotation=False,
            ax=axes[1],
        )


class MCMembership(AttackModel):
    """
    Membership inference attacks by Monte Carlo Membership inference attack.

    See `Hilprecht, Benjamin, Martin Härterich, and Daniel Bernau.
    "Monte Carlo and Reconstruction Membership Inference Attacks against Generative Models."
    Proc. Priv. Enhancing Technol. 2019, no. 4 (2019): 232-249.
    <https://petsymposium.org/popets/2019/popets-2019-0067.pdf>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real dataset
        for better computing performance when evaluating the model
    """

    name = "Monte Carlo Membership"
    alias = "mcmebership"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision_top1%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision_top50%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
        ]
        return submetrics

    def eval(
        self,
        df_test: pd.DataFrame,
        y_test: np.ndarray,
        df_synth: pd.DataFrame,
        cat_cols: list,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate a GAN-Leaks model

        :param df_test: the real data to be evaluated
        :param y_test: the true label of the real data
        :param df_synth: the reference synthetic data
        :param cat_cols: the columns with categorical variables
        :return: top 1% precision and top 50% precision of the predictions and the number of neighbors for each record
        """
        df_synth_ref = df_synth[df_test.columns]  # gower needs the same order

        cat_features = [
            True if col in cat_cols else False for col in df_test.columns
        ]  # boolean array instead of column names

        # Compute Gower distance (adapted to mixed data): data_y is data to compare
        pairwise_gower = gower.gower_matrix(
            data_x=df_test, data_y=df_synth_ref, cat_features=cat_features
        )

        # Fetch the shortest distance for each record
        min_dist = np.min(pairwise_gower, axis=1)

        # Use median heuristic to set the value of epsilon for Ɛ-neighborhood
        eps = np.median(min_dist)

        num_neighbor = np.sum(np.where(pairwise_gower <= eps, 1, 0), axis=1)

        y_pred_proba = num_neighbor / len(df_synth_ref)

        # Compute the metrics
        precision_top1 = self.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba
        )
        precision_top50 = self.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba
        )

        return precision_top1, precision_top50, num_neighbor

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Membership inference attacks based on Ɛ-neighborhood. Evaluate with real data,
        which consists of the real data used to generate 1st generation synthetic data and a control (test set).

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision** and **top 50 precision**
                to predict if a record in the real set (train and test) is used to generate the first generation synthetic data
            * **detailed** -- number of synthetic sample in the Ɛ-neighborhood for each test sample
                **number of neighbors**
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Sample a fraction of the datasets for computation performance.
        # The size of real_train and real_control should be the same, so that the test set is balanced.
        real_control = df_real["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        real_train = df_real["train"].sample(
            n=len(real_control),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        df_test = pd.concat([real_train, real_control], axis=0, ignore_index=True)

        print(f"Monte Carlo Membership test set shape: {df_test.shape}")

        # Label 1 for real records used to generate 1st generation synthetic data and 0 for control
        y_test = np.array([1] * len(real_train) + [0] * len(real_control))

        precision_top1, precision_top50, num_neighbor = self.eval(
            df_test=df_test,
            y_test=y_test,
            df_synth=df_synthetic["train"],
            cat_cols=metadata["categorical"],
        )

        res = {
            "average": {
                "precision_top1%": precision_top1,
                "precision_top50%": precision_top50,
            },
            "detailed": {
                "num_neighbors": num_neighbor,
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a histogram for the number of neighbors in the Ɛ-neighborhood for each test sample

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "num_neighbors",
            ]
        )

        mini = np.min(report["num_neighbors"])
        maxi = np.max(report["num_neighbors"])

        plt.figure(figsize=figsize, layout="constrained")

        udraw.histplot_plot(
            s=pd.Series(report["num_neighbors"]),
            title="MC Membership",
            value_name="# of neighbors",
            stat="count",
            bins=10,
            binrange=(mini, maxi),
            xrotation=False,
        )


class Logan(AttackModel):
    """
    Membership inference attacks by a modified version of LOGAN.

    See `Hayes, Jamie, Luca Melis, George Danezis, and Emiliano De Cristofaro.
    "Logan: Membership inference attacks against generative models."
    arXiv preprint arXiv:1705.07663 (2017) <https://arxiv.org/abs/1705.07663>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real dataset
        for better computing performance when evaluating the model
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "LOGAN"
    alias = "logan"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision_top1%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision_top50%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.001%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.1%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
        ]
        return submetrics

    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        cont_cols: list,
        cat_cols: list,
    ) -> Pipeline:
        """
        Train a LOGAN model

        :param df_train: the train data
        :param y_train: the train label
        :param cont_cols: the columns with continuous variables
        :param cat_cols: the columns with categorical variables
        :return: trained model
        """

        pipe = self.hyperparam_tuning(
            x=df_train,
            y=y_train,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
        )

        return pipe

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Train a model for membership inference attacks. Evaluate the model with real data,
        which consists of the real data used to generate 1st generation synthetic data and a control (test set).
        Output precision and ROC

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision**, **top 50 precision** and **precision**
              to predict if a record in the real set (train and test) is used to generate the first generation synthetic data

            * **detailed** -- **top 1% precision**, **top 50% precision**, **precision** and **ROC** for each repetition
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Transform original dataframes into input and output arrays for the training stage
        df_train = pd.concat(
            [df_synthetic["train"], df_synthetic["2nd_gen"]], axis=0, ignore_index=True
        )

        # Sample a fraction of the datasets for computation performance.
        # The size of real_train and real_control should be the same, so that the test set is balanced.
        real_control = df_real["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        real_train = df_real["train"].sample(
            n=len(real_control),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        df_test = pd.concat([real_train, real_control], axis=0, ignore_index=True)

        print(f"LOGAN test set shape: {df_test.shape}")

        # Select the columns keeping the order
        cat_cols = [
            col for col in df_real["train"].columns if col not in metadata["continuous"]
        ]
        cont_cols = [col for col in df_real["train"].columns if col not in cat_cols]
        df_train[cat_cols] = df_train[cat_cols].astype("object")
        df_test[cat_cols] = df_test[cat_cols].astype("object")

        # Label 1 for 1st generation synthetic data used to generate 2nd generation synthetic data and 0 for 2nd generation sytnthetic data.
        # Label 1 for real records used to generate 1st generation synthetic data and 0 for control.
        y_train = np.array(
            [1] * len(df_synthetic["train"]) + [0] * len(df_synthetic["2nd_gen"])
        )
        y_test = np.array([1] * len(real_train) + [0] * len(real_control))

        # Compute the metrics
        precision_top1 = []
        precision_top50 = []
        precision = []
        tpr_at_lowest_fpr = []  # at 0.001%
        tpr_at_lower_fpr = []  # at 0.1%
        roc = []

        # Compute scores several times to account for randomness
        for _ in range(self._num_repeat):
            pipe = self.fit(
                df_train=df_train,
                y_train=y_train,
                cont_cols=cont_cols,
                cat_cols=cat_cols,
            )

            # Code below to use later if we want to place the test data on GPU to avoid mismatch
            # It will have to be integrated in the tuning as well (with manual cross-validation)
            # if self._use_gpu:
            #    y_pred_proba = ulearning.gpu_predict_proba(pipe, df_test)
            # else:
            y_pred_proba = pipe.predict_proba(df_test)[
                :, 1
            ]  # binary case, y_pred needs to be (num_samples,)

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            tpr_lowest = self.tpr_at_n_fpr(0.001, fpr, tpr)
            tpr_lower = self.tpr_at_n_fpr(0.1, fpr, tpr)
            tpr_at_lowest_fpr.append(tpr_lowest)
            tpr_at_lower_fpr.append(tpr_lower)
            roc.append([fpr, tpr])

            precision_top_1 = self.precision_top_n(
                n=1, y_true=y_test, y_pred_proba=y_pred_proba
            )
            precision_top_50 = self.precision_top_n(
                n=50, y_true=y_test, y_pred_proba=y_pred_proba
            )

            precision_top1.append(precision_top_1)
            precision_top50.append(precision_top_50)

            # Convert probability to class prediction
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)

            prec_score = precision_score(y_test, y_pred)
            precision.append(prec_score)

        res = {
            "average": {
                "precision_top1%": np.mean(precision_top1),
                "precision_top50%": np.mean(precision_top50),
                "precision": np.mean(precision),
                "tpr_at_0.001%_fpr": np.mean(tpr_at_lowest_fpr),
                "tpr_at_0.1%_fpr": np.mean(tpr_at_lower_fpr),
            },
            "detailed": {
                "precision_top1%": np.array(precision_top1),
                "precision_top50%": np.array(precision_top50),
                "precision": np.array(precision),
                "roc": roc,
                "prediction": {
                    "test_index": df_test.index,
                    "y_test": y_test,
                    "model_pred": y_pred_proba,
                },
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the different scores and a log-log graph for ROC

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "precision_top1%",
                "precision_top50%",
                "precision",
                "roc",
            ]
        )

        # Bar plot single value
        plt.figure(figsize=figsize, layout="constrained")

        data = pd.DataFrame(
            {
                "precision_top1%": report["precision_top1%"],
                "precision_top50%": report["precision_top50%"],
                "precision": report["precision"],
            }
        )
        udraw.bar_plot(
            data=data,
            title=f"Metric: {cls.name}",
            value_name="",
        )

        fpr_tpr_list = report["roc"]

        labels_list = []
        for fpr, tpr in fpr_tpr_list:
            auc_roc = auc(fpr, tpr)
            metric_text = "auc=%.3f" % auc_roc
            labels_list.append(metric_text)

        # Plot a log-log graph
        plt.figure(figsize=figsize, layout="constrained")

        udraw.plot_log_scale(
            data=fpr_tpr_list,
            title=f"{cls.name}: ROC Curves",
            labels=labels_list,
            x_label="False Positive Rate",
            y_label="True Positive rate",
        )


class TableGan(AttackModel):
    """
    Membership inference attacks by a modified version of TableGan.

    See `Park, Noseong, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park,
    and Youngmin Kim. "Data synthesis based on generative adversarial networks."
    arXiv preprint arXiv:1806.03384 (2018) <https://arxiv.org/abs/1806.03384>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param sampling_frac: the fraction of data to sample from real dataset
        for better computing performance when evaluating the model
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "TableGan"
    alias = "tablegan"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision_top1%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision_top50%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.001%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.1%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
        ]
        return submetrics

    def fit(
        self,
        df_train_discriminator: pd.DataFrame,
        y_train_discriminator: np.ndarray,
        df_train_classifier: pd.DataFrame,
        y_train_classifier: np.ndarray,
        cont_cols: list,
        cat_cols: list,
    ) -> Tuple[Pipeline, Pipeline]:
        """
        Train a TableGAN model

        :param df_train_discriminator: the train data for discriminator
        :param y_train_discriminator: the train label for discriminator
        :param df_train_classifier: the train data for classifier
        :param y_train_classifier: the train label for classifier
        :param cont_cols: the columns with continuous variables
        :param cat_cols: the columns with categorical variables
        :return: trained discriminator and classifier
        """

        # Train the discriminator
        pipe_discriminator = self.hyperparam_tuning(
            x=df_train_discriminator,
            y=y_train_discriminator,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
        )

        y_pred_proba_discriminator = pipe_discriminator.predict_proba(
            df_train_classifier
        )[:, 1]

        # Train the final classifier
        df_train_classifier_copy = df_train_classifier.copy()
        df_train_classifier_copy["score"] = y_pred_proba_discriminator.tolist()

        pipe_classifier = self.hyperparam_tuning(
            x=df_train_classifier_copy,
            y=y_train_classifier,
            continuous_cols=cont_cols + ["score"],
            categorical_cols=cat_cols,
        )

        return pipe_discriminator, pipe_classifier

    def pred_proba(
        self,
        df: pd.DataFrame,
        trained_discriminator: Pipeline,
        trained_classifier: Pipeline,
    ) -> np.ndarray:
        """
        Predict the probability with a trained TableGAN model

        :param df: input data
        :param trained_discriminator: trained discriminator
        :param trained_classifier: trained classifier
        :return: the predicted probability
        """
        score = trained_discriminator.predict_proba(df)[:, 1]
        df_copy = df.copy()
        df_copy["score"] = score.tolist()

        y_pred_proba = trained_classifier.predict_proba(df_copy)[:, 1]

        return y_pred_proba

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Train a model for membership inference attacks. Evaluate the model with real data,
        which consists of the real data used to generate 1st generation synthetic data and a control (test set).
        Output precision and ROC.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision**, **top 50 precision** and **precision**
              to predict if a record in the real set (train and test) is used to generate the first generation synthetic data

            * **detailed** -- **top 1% precision**, **top 50% precision**, **precision** and **ROC** for each repetition
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Split the 1st generation synthetic train set into 2 sets: 1 used to train discriminator and another used to train final classifier
        df_synth_train_classifier = df_synthetic["train"].sample(
            n=len(df_synthetic["test"]),
            replace=False,
            ignore_index=False,
        )
        df_synth_train_discriminator = df_synthetic["train"][
            ~df_synthetic["train"].index.isin(df_synth_train_classifier.index)
        ].reset_index(drop=True)

        # Construct train set to train the discriminator: 1st gen + 2nd gen synthetic sets
        df_synth_2nd_gen = df_synthetic["2nd_gen"].sample(
            n=len(df_synth_train_discriminator),
            replace=False,
            ignore_index=True,
        )
        df_train_discriminator = pd.concat(
            [df_synth_train_discriminator, df_synth_2nd_gen],
            axis=0,
            ignore_index=True,
        )

        # Construct the train set used to train the final classifier, which contains 1st generation of
        # synthetic data which is used to generate the 2nd generation synthetic data and control set
        df_train_classifier = pd.concat(
            [df_synth_train_classifier.reset_index(drop=True), df_synthetic["test"]],
            axis=0,
            ignore_index=True,
        )

        # Construct the test set for the final classifier
        # Sample a fraction of the datasets for computation performance
        # The size of real_train and real_control should be the same, so that the test set is balanced
        real_control = df_real["test"].sample(
            frac=self._sampling_frac,
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        real_train = df_real["train"].sample(
            n=len(real_control),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        df_test = pd.concat([real_train, real_control], axis=0, ignore_index=True)

        print(f"TableGan test set shape: {df_test.shape}")

        # Select the columns keeping the order
        cat_cols = [
            col for col in df_real["train"].columns if col not in metadata["continuous"]
        ]
        cont_cols = [col for col in df_real["train"].columns if col not in cat_cols]
        df_train_discriminator[cat_cols] = df_train_discriminator[cat_cols].astype(
            "object"
        )
        df_train_classifier[cat_cols] = df_train_classifier[cat_cols].astype("object")
        df_test[cat_cols] = df_test[cat_cols].astype("object")

        # Encode labels

        # Label 1 for 1st generation synthetic data and 0 for 2nd generation synthetic data.
        y_train_discriminator = np.array(
            [1] * len(df_synth_train_discriminator) + [0] * len(df_synth_2nd_gen)
        )

        # Label 1 for 1st gen synthetic data used to generate 2nd generation synthetic data and 0 for control.
        y_train_classifier = np.array(
            [1] * len(df_synth_train_classifier) + [0] * len(df_synthetic["test"])
        )

        # Label 1 for real records used to generate 1st generation synthetic data and 0 for control.
        y_test = np.array([1] * len(real_train) + [0] * len(real_control))

        # Compute the metrics
        precision_top1 = []
        precision_top50 = []
        precision = []
        tpr_at_lowest_fpr = []  # at 0.001%
        tpr_at_lower_fpr = []  # at 0.1%
        roc = []

        # Compute scores several times to account for randomness
        for _ in range(self._num_repeat):
            pipe_discriminator, pipe_classifier = self.fit(
                df_train_discriminator=df_train_discriminator,
                y_train_discriminator=y_train_discriminator,
                df_train_classifier=df_train_classifier,
                y_train_classifier=y_train_classifier,
                cont_cols=cont_cols,
                cat_cols=cat_cols,
            )

            y_test_pred_proba = self.pred_proba(
                df=df_test,
                trained_discriminator=pipe_discriminator,
                trained_classifier=pipe_classifier,
            )

            fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
            tpr_lowest = self.tpr_at_n_fpr(0.001, fpr, tpr)
            tpr_lower = self.tpr_at_n_fpr(0.1, fpr, tpr)
            tpr_at_lowest_fpr.append(tpr_lowest)
            tpr_at_lower_fpr.append(tpr_lower)
            roc.append([fpr, tpr])

            precision_top_1 = self.precision_top_n(
                n=1, y_true=y_test, y_pred_proba=y_test_pred_proba
            )
            precision_top_50 = self.precision_top_n(
                n=50, y_true=y_test, y_pred_proba=y_test_pred_proba
            )

            precision_top1.append(precision_top_1)
            precision_top50.append(precision_top_50)

            # Convert probability to class prediction
            y_test_pred = np.where(y_test_pred_proba > 0.5, 1, 0)

            prec_score = precision_score(y_test, y_test_pred)
            precision.append(prec_score)

        res = {
            "average": {
                "precision_top1%": np.mean(precision_top1),
                "precision_top50%": np.mean(precision_top50),
                "precision": np.mean(precision),
                "tpr_at_0.001%_fpr": np.mean(tpr_at_lowest_fpr),
                "tpr_at_0.1%_fpr": np.mean(tpr_at_lower_fpr),
            },
            "detailed": {
                "precision_top1%": np.array(precision_top1),
                "precision_top50%": np.array(precision_top50),
                "precision": np.array(precision),
                "roc": roc,
                "prediction": {
                    "test_index": df_test.index,
                    "y_test": y_test,
                    "model_pred": y_test_pred_proba,
                },
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the different scores and a log-log graph for ROC

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "precision_top1%",
                "precision_top50%",
                "precision",
                "roc",
            ]
        )

        # Bar plot single value
        plt.figure(figsize=figsize, layout="constrained")

        data = pd.DataFrame(
            {
                "precision_top1%": report["precision_top1%"],
                "precision_top50%": report["precision_top50%"],
                "precision": report["precision"],
            }
        )
        udraw.bar_plot(
            data=data,
            title=f"Metric: {cls.name}",
            value_name="",
        )

        fpr_tpr_list = report["roc"]

        labels_list = []
        for fpr, tpr in fpr_tpr_list:
            auc_roc = auc(fpr, tpr)
            metric_text = "auc=%.3f" % auc_roc
            labels_list.append(metric_text)

        # Plot a log-log graph
        plt.figure(figsize=figsize, layout="constrained")

        udraw.plot_log_scale(
            data=fpr_tpr_list,
            title=f"{cls.name}: ROC Curves",
            labels=labels_list,
            x_label="False Positive Rate",
            y_label="True Positive rate",
        )


class Detector(AttackModel):
    """
    Membership inference attacks by a modified version of Detector Networks.

    See `Olagoke, Lukman, Salil Vadhan, and Seth Neel.
    "Black-Box Training Data Identification in GANs via Detector Networks."
    arXiv preprint arXiv:2310.12063 (2023). <https://arxiv.org/abs/2310.12063>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Detector"
    alias = "detector"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision_top1%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision_top50%",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "precision",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.001%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "tpr_at_0.1%_fpr",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
        ]
        return submetrics

    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        cont_cols: list,
        cat_cols: list,
    ) -> Pipeline:
        """
        Train a Detector model

        :param df_train: the train data
        :param y_train: the train label
        :param cont_cols: the columns with continuous variables
        :param cat_cols: the columns with categorical variables
        :return: trained model
        """

        pipe = self.hyperparam_tuning(
            x=df_train,
            y=y_train,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
        )

        return pipe

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Train a model for membership inference attacks. Evaluate the model with real data,
        which consists of the real data used to generate 1st generation synthetic data and a control (test set).
        Output precision and ROC.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision**, **top 50 precision** and **precision**
              to predict if a record in the real set (train and test) is used to generate the first generation synthetic data

            * **detailed** -- **top 1% precision**, **top 50% precision**, **precision** and **ROC** for each repetition
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Split the real test data which is not used to generate 1st generation synthetic data into 2 sets:
        # 1 set used to train the detector and another set for control (ratio = 80%:20%)
        real_control = df_real["test"].sample(
            frac=0.5,
            replace=False,
            ignore_index=False,
            random_state=self._random_state,
        )
        real_train_detector = df_real["test"][
            ~df_real["test"].index.isin(real_control.index)
        ].reset_index(drop=True)

        # Sample from 1st generation synthetic data to train detector
        synth_train_detector = df_synthetic["train"].sample(
            n=len(real_train_detector),
            replace=False,
            ignore_index=True,
        )

        # Construct the train set to train the detector
        df_train = pd.concat(
            [real_train_detector, synth_train_detector],
            axis=0,
            ignore_index=True,
        )

        # Sample from the real data used to generate 1st generation synthetic data to be used as part of the test set
        # The size of real_train and real_control should be the same, so that the test set is balanced.
        real_train = df_real["train"].sample(
            n=len(real_control),
            replace=False,
            ignore_index=True,
            random_state=self._random_state,
        )

        # Construct the test set
        df_test = pd.concat(
            [real_train, real_control.reset_index(drop=True)], axis=0, ignore_index=True
        )

        print(f"Detector test set shape: {df_test.shape}")

        # Select the columns keeping the order
        cat_cols = [
            col for col in df_real["train"].columns if col not in metadata["continuous"]
        ]
        cont_cols = [col for col in df_real["train"].columns if col not in cat_cols]
        df_train[cat_cols] = df_train[cat_cols].astype("object")
        df_test[cat_cols] = df_test[cat_cols].astype("object")

        # Train set: label 1 for generated synthetic data 0 for reference fresh real data.
        y_train = np.array(
            [0] * len(real_train_detector) + [1] * len(synth_train_detector)
        )

        # Test set: label 1 for real records used to generate 1st generation synthetic data and 0 for control.
        y_test = np.array([1] * len(real_train) + [0] * len(real_control))

        # Compute the metrics
        precision_top1 = []
        precision_top50 = []
        precision = []
        tpr_at_lowest_fpr = []  # at 0.001%
        tpr_at_lower_fpr = []  # at 0.1%
        roc = []

        # Compute scores several times to account for randomness
        for _ in range(self._num_repeat):
            pipe = self.fit(
                df_train=df_train,
                y_train=y_train,
                cont_cols=cont_cols,
                cat_cols=cat_cols,
            )

            y_pred_proba = pipe.predict_proba(df_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            tpr_lowest = self.tpr_at_n_fpr(0.001, fpr, tpr)
            tpr_lower = self.tpr_at_n_fpr(0.1, fpr, tpr)
            tpr_at_lowest_fpr.append(tpr_lowest)
            tpr_at_lower_fpr.append(tpr_lower)
            roc.append([fpr, tpr])

            precision_top_1 = self.precision_top_n(
                n=1, y_true=y_test, y_pred_proba=y_pred_proba
            )
            precision_top_50 = self.precision_top_n(
                n=50, y_true=y_test, y_pred_proba=y_pred_proba
            )

            precision_top1.append(precision_top_1)
            precision_top50.append(precision_top_50)

            # Convert probability to class prediction
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)

            prec_score = precision_score(y_test, y_pred)
            precision.append(prec_score)

        res = {
            "average": {
                "precision_top1%": np.mean(precision_top1),
                "precision_top50%": np.mean(precision_top50),
                "precision": np.mean(precision),
                "tpr_at_0.001%_fpr": np.mean(tpr_at_lowest_fpr),
                "tpr_at_0.1%_fpr": np.mean(tpr_at_lower_fpr),
            },
            "detailed": {
                "precision_top1%": np.array(precision_top1),
                "precision_top50%": np.array(precision_top50),
                "precision": np.array(precision),
                "roc": roc,
                "prediction": {
                    "test_index": df_test.index,
                    "y_test": y_test,
                    "model_pred": y_pred_proba,
                },
            },
        }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the different scores and a log-log graph for ROC

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "precision_top1%",
                "precision_top50%",
                "precision",
                "roc",
            ]
        )

        # Bar plot single value
        plt.figure(figsize=figsize, layout="constrained")

        data = pd.DataFrame(
            {
                "precision_top1%": report["precision_top1%"],
                "precision_top50%": report["precision_top50%"],
                "precision": report["precision"],
            }
        )
        udraw.bar_plot(
            data=data,
            title=f"Metric: {cls.name}",
            value_name="",
        )

        fpr_tpr_list = report["roc"]

        labels_list = []
        for fpr, tpr in fpr_tpr_list:
            auc_roc = auc(fpr, tpr)
            metric_text = "auc=%.3f" % auc_roc
            labels_list.append(metric_text)

        # Plot a log-log graph
        plt.figure(figsize=figsize, layout="constrained")

        udraw.plot_log_scale(
            data=fpr_tpr_list,
            title=f"{cls.name}: ROC Curves",
            labels=labels_list,
            x_label="False Positive Rate",
            y_label="True Positive rate",
        )


class Collision(AttackModel):
    """
    Modified version of membership collision attack.

    See `Hu, Aoting, Renjie Xie, Zhigang Lu, Aiqun Hu, and Minhui Xue.
    "TableGAN-MCA: Evaluating membership collisions of GAN-synthesized tabular data releasing."
    In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security,
    pp. 2096-2112. 2021. <https://dl.acm.org/doi/abs/10.1145/3460120.3485251>`_
    for more details.

    :cvar name: the name of the attack model
    :vartype name: str
    :cvar alias: the shortname of the attack model
    :vartype alias: str


    :param random_state: for reproducibility purposes
    :param num_repeat: the scores are averaged across the number of repetitions to account for randomness
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    """

    name = "Collision"
    alias = "collision"

    @classmethod
    def get_average_submetrics(cls) -> List[dict]:
        """
        Get the average submetrics of the current metric with their target and min/max values.

        :return: the list of the average submetrics
        """

        submetrics = [
            {
                "submetric": "precision",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "recall",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "f1_score",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "recovery_rate",
                "min": 0,
                "max": 1.0,
                "objective": "min",
            },
            {
                "submetric": "avg_num_appearance_realtrain",
                "min": 0,
                "max": np.inf,
                "objective": "-",
            },
            {
                "submetric": "avg_num_appearance_realcontrol",
                "min": 0,
                "max": np.inf,
                "objective": "-",
            },
            {
                "submetric": "avg_num_appearance_synth",
                "min": 0,
                "max": np.inf,
                "objective": "-",
            },
            {
                "submetric": "avg_num_appearance_collision_real",
                "min": 0,
                "max": np.inf,
                "objective": "-",
            },
            {
                "submetric": "avg_num_appearance_collision_synth",
                "min": 0,
                "max": np.inf,
                "objective": "-",
            },
        ]
        return submetrics

    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        cont_cols: list,
        cat_cols: list,
    ) -> Pipeline:
        """
        Train a Collision attack model

        :param df_train: the train data with occurrence frequency added as extra feature
        :param y_train: the train label
        :param cont_cols: the columns with continuous variables including occurrence frequency
        :param cat_cols: the columns with categorical variables
        :return: trained model
        """

        pipe = self.hyperparam_tuning(
            x=df_train,
            y=y_train,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
        )

        return pipe

    def compute(
        self,
        df_real: dict[str, pd.DataFrame],
        df_synthetic: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Train a model for membership collision attacks.
        Evaluate the model with real data and 1st generation synthetic data.

        :param df_real: the real dataset, split into **train** and **test** sets
        :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
        :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
        :return: a dictionary with two keys pointing to dictionaries

            * **average** -- the average across repetitions for **top 1% precision**, **top 50 precision** and **precision**
              to predict if a record in the real set (train and test) is used to generate the first generation synthetic data
              and the average number of appearance of each unique record in real train, real control, synthetic set
              and for collisions

            * **detailed** -- **top 1% precision**, **top 50% precision**, **precision** and **ROC** for each repetition
        """

        self.check_consistency_compute_parameters(df_real, df_synthetic, metadata)

        if df_synthetic["train"].shape[1] <= 1:
            return {}

        # Compute the number of appearance of each unique record in different set
        len_real_train = len(df_real["train"])
        avg_num_appearance_realtrain = np.mean(
            self.compute_frequency(df_real["train"])["frequency"] * len_real_train
        )

        len_real_control = len(df_real["test"])
        avg_num_appearance_realcontrol = np.mean(
            self.compute_frequency(df_real["test"])["frequency"] * len_real_control
        )

        len_synth = len(df_synthetic["train"])
        avg_num_appearance_synth = np.mean(
            self.compute_frequency(df_synthetic["train"])["frequency"] * len_synth
        )

        # Compute the number of appearance of each unique record for only collisions
        real_notation = self.annotate_collision(
            df_target=df_real["train"],
            df_ref=df_synthetic["train"],
        )
        real_collision = real_notation.copy()[real_notation["collision"] == 1]
        len_real_collision = len(real_collision)
        avg_num_appearance_collision_real = np.mean(
            self.compute_frequency(real_collision)["frequency"] * len_real_collision
        )

        synth_notation = self.annotate_collision(
            df_target=df_synthetic["train"],
            df_ref=df_real["train"],
        )
        synth_collision = synth_notation.copy()[synth_notation["collision"] == 1]
        len_synth_collision = len(synth_collision)
        avg_num_appearance_collision_synth = np.mean(
            self.compute_frequency(synth_collision)["frequency"] * len_synth_collision
        )

        # Add frequency to the 2nd generation synthetic data
        df_train = self.compute_frequency(df_synthetic["2nd_gen"])

        # Find collisions in the 2nd generation synthetic data: records that also appear in the 1st generation synthetic data used to generate 2nd generation synthetic data
        y_train = np.array(
            self.annotate_collision(
                df_target=df_train.drop("frequency", axis=1),
                df_ref=df_synthetic["train"],
            )["collision"]
        )

        # Add frequency to 1st generation synthetic data (test set)
        df_test = self.compute_frequency(df_synthetic["train"])

        y_test = np.array(
            self.annotate_collision(
                df_target=df_test.drop("frequency", axis=1),
                df_ref=df_real["train"],
            )["collision"]
        )

        # Select the columns keeping the order
        cat_cols = [
            col for col in df_real["train"].columns if col not in metadata["continuous"]
        ]
        cont_cols = [col for col in df_real["train"].columns if col not in cat_cols]
        df_train[cat_cols] = df_train[cat_cols].astype("object")
        df_test[cat_cols] = df_test[cat_cols].astype("object")

        precision = []
        recall = []
        f1 = []
        recovery_rate = []
        pr_curve = []

        if (
            (len(y_train[y_train == 0]) < self._num_kfolds)
            | (len(y_train[y_train == 1]) < self._num_kfolds)
            | (len(y_test[y_test == 0]) == 0)
            | (len(y_test[y_test == 1]) == 0)
        ):
            # If there's no collision
            res = {
                "average": {
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "recovery_rate": np.nan,
                    "avg_num_appearance_realtrain": avg_num_appearance_realtrain,
                    "avg_num_appearance_realcontrol": avg_num_appearance_realcontrol,
                    "avg_num_appearance_synth": avg_num_appearance_synth,
                    "avg_num_appearance_collision_real": np.nan,
                    "avg_num_appearance_collision_synth": np.nan,
                },
                "detailed": {
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "recovery_rate": np.nan,
                    "pr_curve": np.nan,
                },
            }
        else:
            # Compute scores several times to account for randomness
            for _ in range(self._num_repeat):
                pipe = self.fit(
                    df_train=df_train,
                    y_train=y_train,
                    cont_cols=cont_cols + ["frequency"],
                    cat_cols=cat_cols,
                )

                y_pred_proba = pipe.predict_proba(df_test)[:, 1]

                # Calculate precision-recall curve
                p, r, th = precision_recall_curve(y_test, y_pred_proba)
                pr_curve.append([p, r])

                # Convert probability to class prediction and set the threshold to lower value
                y_pred = np.where(y_pred_proba > 0.1, 1, 0)

                precision_score_ = precision_score(y_test, y_pred)
                recall_score_ = recall_score(y_test, y_pred)
                f1_ = f1_score(y_test, y_pred)

                # Calculate recovery rate
                recovered_train = recall_score_ * len(y_test[y_test == 1])
                recovery_rate_ = recovered_train / len(
                    df_real["train"].drop_duplicates()
                )

                precision.append(precision_score_)
                recall.append(recall_score_)
                f1.append(f1_)
                recovery_rate.append(recovery_rate_)

            res = {
                "average": {
                    "precision": np.mean(precision),
                    "recall": np.mean(recall),
                    "f1_score": np.mean(f1),
                    "recovery_rate": np.mean(recovery_rate),
                    "avg_num_appearance_realtrain": avg_num_appearance_realtrain,
                    "avg_num_appearance_realcontrol": avg_num_appearance_realcontrol,
                    "avg_num_appearance_synth": avg_num_appearance_synth,
                    "avg_num_appearance_collision_real": avg_num_appearance_collision_real,
                    "avg_num_appearance_collision_synth": avg_num_appearance_collision_synth,
                },
                "detailed": {
                    "precision": np.array(precision),
                    "recall": np.array(recall),
                    "f1_score": np.array(f1),
                    "recovery_rate": np.array(recovery_rate),
                    "pr_curve": pr_curve,
                },
            }

        return res

    @classmethod
    def draw(cls, report: dict, figsize: Tuple[float, float] = None) -> None:
        """
        Draw a barplot to compare the different scores and a line plot for precision-recall curve

        :param report: the **detailed** report, outcome of the *compute* method
        :param figsize: the size of the figure in inches (width, height)
        :return: *None*
        """
        assert report is not None
        assert all(
            key in report
            for key in [
                "precision",
                "recall",
                "f1_score",
                "recovery_rate",
                "pr_curve",
            ]
        )

        if all(np.all(np.isnan(value)) for value in report.values()):
            pass
        else:
            # Bar plot single value
            plt.figure(figsize=figsize, layout="constrained")

            data = pd.DataFrame(
                {
                    "precision": report["precision"],
                    "recall": report["recall"],
                    "f1_score": report["f1_score"],
                    "recovery_rate": report["recovery_rate"],
                }
            )
            udraw.bar_plot(
                data=data,
                title=f"Metric: {cls.name}",
                value_name="",
            )

            pr_list = report["pr_curve"]

            plt.figure(figsize=figsize, layout="constrained")

            udraw.line_plot(
                data=pr_list,
                title=f"{cls.name}: precision-recall curve",
                x_label="Recall",
                y_label="Precision",
            )
