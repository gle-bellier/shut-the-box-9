import pickle
from typing import Any, List


def read_pickle(path: str) -> Any:
    """Read pickle file.
    Args:
        path (str): path to the pickle file.
    Returns:
        Any: data from the pickle file.
    Yields:
        Iterator[Any]: iterator on the data.
    """ ""
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)

        except EOFError:
            pass


def write_pickle(data: dict or List[dict], path: str) -> None:
    """Export data into pickle file and overwrite if existing.
    Args:
        data (dict): data dictionary
        path (str): path to the file
    """
    if isinstance(data, dict):
        data = [data]

    with open(path, "wb") as file_out:
        for elt in data:
            pickle.dump(elt, file_out)


def add_to_pickle(data: dict or List[dict], path: str) -> None:
    """Export data into pickle file that can already exist.
    Args:
        data (dict): data dictionary
        path (str): path to the file
    """
    if isinstance(data, dict):
        data = [data]

    with open(path, "ab+") as file_out:
        for elt in data:
            pickle.dump(elt, file_out)