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

    :param filepath: the filename of the object with the extension
    :return: the pickled object
    """

    path = Path(filepath)
    assert path.exists()
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_pickle(
    obj: Any, folderpath: Union[Path, str], filename: str, date: bool = False
) -> None:
    """
    Save an object with pickle.

    :param obj: the object to pickle
    :param folderpath: the path of the folder where to save the object
    :param filename: the filename of the object to pickle without the extension
    :param date: add the date at the beginning of the filename if *True*
    :return: *None*
    """

    path = Path(folderpath)
    assert path.exists()
    if obj is not None:
        name = f"{get_date()}_{filename}" if date else filename
        with open(path / f"{name}.pkl", "wb") as file:
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


def create_directory(path: Union[Path, str]) -> None:
    """
    Create directory if it does not exist

    :param path: the directory to be created
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
