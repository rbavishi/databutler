import json
import os
import re
from typing import Dict

import requests

from databutler.utils import paths


def get_working_dir_for_mining() -> str:
    """
    Returns the path of the directory to use for downloading and running notebooks.

    Returns:
        (str): A string corresponding to the path.
    """
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "mining", "kaggle")


def get_notebook_data(username: str, kernel_slug: str) -> Dict:
    """
    Fetches all the metadata, including sources, for a notebook given the username of the owner and the kernel slug.

    This uses requests to fetch the notebook using URL and uses some regex to get the notebook JSON out.

    Args:
        username (str): A string for the username of the owner.
        kernel_slug: A string for the slug of the kernel.

    Returns:
        (Dict): A dictionary containing the notebook data.
    """
    text = requests.get(f"https://www.kaggle.com/{username}/{kernel_slug}").text
    #  NOTE: Although this is not too likely to change, this method will break if Kaggle decides to change their
    #  website structure.
    d = next(i for i in re.compile(r"Kaggle.State.push\(({.*})\)").findall(text) if "runInfo" in i)
    d = json.loads(d)

    return d
