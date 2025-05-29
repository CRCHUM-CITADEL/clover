# Standard library
from typing import List, Tuple, Union
from inspect import getfullargspec
from pathlib import Path

# 3rd party packages
import pandas as pd

# Local packages
from .utility.report import UtilityReport
from .privacy.report import PrivacyReport


class Report:
    """
    Create a combined report of the metrics, whether they are utility or privacy metrics.

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param df_synthetic: the synthetic dataset, split into **train**, **test** and **2nd_gen** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param report_folderpath: the path of computed utility and/or privacy reports if available
    :param report_filename: the name of the report (without extension nor utility/privacy) if available
    :param metrics: list of the metrics to compute. Can be utility or privacy metrics.
        If not specified, all the metrics are computed.
    :param params: the dictionary containing the parameters for both utility and privacy reports
    """

    def __init__(
        self,
        dataset_name: str = None,
        df_real: dict[str, pd.DataFrame] = None,
        df_synthetic: dict[str, pd.DataFrame] = None,
        metadata: dict = None,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
        report_folderpath: Union[Path, str] = None,
        report_filename: str = None,
        metrics: List[str] = None,
        params: dict = None,
    ):
        assert report_folderpath is None or Path(report_folderpath).exists()
        assert (report_folderpath is not None and report_filename is not None) or all(
            v is not None for v in [dataset_name, df_real, df_synthetic, metadata]
        )
        available_metrics = self.__class__.get_available_metrics()
        assert metrics is None or (
            len(metrics) > 0 and all(m in available_metrics for m in metrics)
        ), f"metrics should be None or a list of names if the following list: {available_metrics}"

        self._reports = {}

        common_params = {
            "dataset_name": dataset_name,
            "df_real": df_real,
            "df_synthetic": df_synthetic,
            "metadata": metadata,
            "figsize": figsize,
            "random_state": random_state,
        }

        for name, report in zip(["utility", "privacy"], [UtilityReport, PrivacyReport]):
            # Check if the metrics are computed in the Utility or Privacy report or both
            available_metrics = set(report.metrics_mapping.keys())
            metrics_to_compute = (
                set(metrics).intersection(available_metrics)
                if metrics is not None
                else available_metrics
            )

            if report_folderpath is None:
                if (
                    params is not None
                    and "cross_learning" in params
                    and params["cross_learning"] == False
                ):
                    metrics_to_compute.discard("Cross Classification")
                    metrics_to_compute.discard("Cross Regression")

                if len(metrics_to_compute) > 0:
                    other_params = params if params is not None else {}
                    self._reports[name] = report(  # instantiate the report
                        **common_params,
                        **{
                            arg: other_params[arg]
                            for arg in getfullargspec(report).args[1:]  # remove self
                            if arg in other_params.keys()
                        },
                        metrics=metrics_to_compute,
                    )

            else:
                path = Path(report_folderpath) / f"{report_filename}_{name}.pkl"
                if path.exists():
                    self._reports[name] = report(report_filepath=path)

        assert len(self._reports) > 0

    @classmethod
    def get_available_metrics(cls) -> list[str]:
        """
        Get the available list of metrics

        :return: the available metrics name
        """

        utility_metrics = list(UtilityReport.metrics_mapping.keys())
        privacy_metrics = list(PrivacyReport.metrics_mapping.keys())

        return utility_metrics + privacy_metrics

    @classmethod
    def get_metrics_info(cls) -> pd.DataFrame:
        """
        Get the average submetrics information for all the utility and privacy metrics.

        :return: the submetrics information as a dataframe
        """
        metrics_info = []

        for report in [UtilityReport, PrivacyReport]:
            metrics_info.append(report.get_metrics_info())

        df = pd.concat(metrics_info, axis=0)

        return df

    def compute(self) -> None:
        """
        Compute the reports.

        :return: *None*
        """

        for name in self._reports:
            self._reports[name].compute()

    def specification(self) -> None:
        """
        Print the dataset specification.

        :return: *None*
        """

        if len(self._reports) > 0:
            self._reports[next(iter(self._reports))].specification()

    def save(self, savepath: Union[Path, str], filename: str) -> None:
        """
        Pickle the utility and privacy reports separately.

        :param savepath: the save folder
        :param filename: the filename without the extension. Nb: "utility" and "privacy" are automatically added
        :return: *None*
        """
        for name in self._reports:
            self._reports[name].save(savepath, filename + "_" + name)

    def summary(self) -> pd.DataFrame:
        """
        Report the average metrics values on both utility and privacy aspects.

        :return: a pandas dataframe
        """

        res = []

        for name in self._reports:
            res.append(self._reports[name].summary())

        df = pd.concat(res, axis=0)

        return df
