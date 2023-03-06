from typing import Union, Any  # standard library
from pathlib import Path
import pickle
from datetime import datetime
import os
import sys


def get_date() -> str:
    """
    Get the current date formatted as YYYY-MM-DD.

    :return: the date
    """
    return datetime.today().strftime("%Y-%m-%d")


def load_pickle(filepath: Union[Path, str]) -> Any:
    """
    Load a pickled object.

    :param filepath: the filepath of the pickle object to load
    :return: the pickled object
    """

    with open(Path(filepath), "rb") as file:
        obj = pickle.load(file)
    return obj


def save_pickle(obj: Any, path: Union[Path, str], filename: str) -> None:
    """
    Save an object with pickle.

    :param obj: the object to pickle
    :param path: the path to save the object
    :param filename: the filename of the object to pickle
    :return: *None*
    """

    if obj is not None:
        with open(Path(path) / f"{get_date()}_{filename}.pkl", "wb") as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


class HiddenPrints:
    """
    Block print calls.
    From https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print

    Usage:
        with HiddenPrints():
            print("This will not be printed")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
