# Standard library
from typing import List, Tuple, Union
from pathlib import Path

# 3rd party packages
import pandas as pd
from pandas.io.formats.style import Styler

# Local packages
from .report import Report


class Metareport:
    """
    Create a metareport comparing synthetic datasets with respect to a list of metrics.

    :param dataset_name: the name of the dataset
    :param df_real: the real dataset, split into **train** and **test** sets
    :param synthetic_datasets: synthetic datasets to compare, with each synthetic dataset as a dictionary
        split into **train**, **test** and **2nd_gen** sets
    :param metadata: dictionary with two entries: the **continuous** and **categorical** lists of variables.
        Must be specified by the user since the variable type might be equivocal.
    :param figsize: the size of the figure in inches (width, height)
    :param random_state: for reproducibility purposes
    :param metareport_folderpath: a dictionary containing the path of each computed report to compare
    :param metrics: list of the metrics to compute. Can be utility or privacy metrics.
        If not specified, all the metrics are computed.
    :param params: the dictionary containing the parameters for the report
    """

    def __init__(
        self,
        dataset_name: str = None,
        df_real: dict[str, pd.DataFrame] = None,
        synthetic_datasets: dict[str, dict[str, pd.DataFrame]] = None,
        metadata: dict = None,
        figsize: Tuple[float, float] = (8, 6),
        random_state: int = 0,
        metareport_folderpath: dict[str, Union[Path, str]] = None,
        metrics: List[str] = None,
        params: dict = None,
    ):
        assert metareport_folderpath is None or all(
            Path(metareport_folderpath[name]).exists()
            for name in metareport_folderpath.keys()
        )
        assert metareport_folderpath is not None or all(
            v is not None for v in [dataset_name, df_real, synthetic_datasets, metadata]
        )

        if metareport_folderpath is None:
            assert len(synthetic_datasets) > 0, "No synthetic dataset to compare"

            # Instantiate reports
            self._metareport = {}
            for name in synthetic_datasets:
                # Load each synthetic dataset
                df_synth = synthetic_datasets[name]
                assert len(df_synth["train"]) + len(df_synth["test"]) == len(
                    df_real["train"]
                ) + len(
                    df_real["test"]
                ), "The number of synthetic data samples must be the same that the real set"

                self._metareport[name] = Report(
                    dataset_name,
                    df_real,
                    df_synth,
                    metadata,
                    figsize,
                    random_state,
                    metrics=metrics,
                    params=params,
                )
        else:
            # Instantiate reports
            self._metareport = {}
            for name in metareport_folderpath:
                self._metareport[name] = Report(
                    report_folderpath=metareport_folderpath[name], report_filename=name
                )

    def save(self, savepath: Union[Path, str]) -> None:
        """
        Pickle the metareport, saving each report separately.

        :param savepath: the save folder
        :return: *None*
        """
        for report in self._metareport:
            self._metareport[report].save(savepath, filename=report)

    def compute(self) -> None:
        """
        Compute all reports and store them.

        :return: *None*
        """

        for report in self._metareport:
            self._metareport[report].compute()

    def summary(self) -> pd.DataFrame:
        """
        Create a cross-table of the metrics average values for each compared synthetic dataset.

        :return: a pandas dataframe
        """

        future_df = []
        for report in self._metareport:
            res = self._metareport[report].summary()
            res["compared"] = report
            res["metric"] = res["alias"] + "-" + res["submetric"]
            future_df.append(res[["compared", "metric", "value"]])

        df = pd.concat(future_df, axis=0).pivot(
            index="metric", columns="compared", values="value"
        )

        return df

    @classmethod
    def get_objective(cls) -> dict:
        """
        Get the objective for each metric, whether they are utility or privacy metrics.

        :return: a dictionary containing the subset of metrics to maximise **max** and the ones to minimize **min**
        """

        objective = {}

        info = Report.get_metrics_info()
        info["metric"] = info["alias"] + "-" + info["submetric"]

        objective["min"] = list(info.query("objective=='min'")["metric"])
        objective["max"] = list(info.query("objective=='max'")["metric"])

        return objective

    @staticmethod
    def make_pretty(styler: Styler, metrics: List[str]) -> Styler:
        """
        Color in green the metric closer to the objective for each row and in yellow the farthest from the objective.

        :param styler: the styler object to format
        :param metrics: the list of the metrics in the index
        :return: the styler formatted
        """

        objective = Metareport.get_objective()
        for key in ["min", "max"]:
            objective[key] = [m for m in objective[key] if m in metrics]

        styler.format(precision=2).highlight_min(
            subset=pd.IndexSlice[objective["max"], :], color="lemonchiffon", axis=1
        ).highlight_max(
            subset=pd.IndexSlice[objective["min"], :], color="lemonchiffon", axis=1
        ).highlight_max(
            subset=pd.IndexSlice[objective["max"], :], color="mediumaquamarine", axis=1
        ).highlight_min(
            subset=pd.IndexSlice[objective["min"], :], color="mediumaquamarine", axis=1
        )

        return styler
