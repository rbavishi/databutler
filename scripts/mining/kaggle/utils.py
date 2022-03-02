import os

from databutler.utils import paths


def get_working_dir_for_mining() -> str:
    """
    Returns the path of the directory to use for downloading and running notebooks.

    Returns:
        (str): A string corresponding to the path.
    """
    path = os.path.join(paths.get_user_home_dir_path(), ".databutler", "mining", "kaggle")
    os.makedirs(path, exist_ok=True)
    return path
