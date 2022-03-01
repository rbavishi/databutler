import json
import os
import re
from enum import Enum
from typing import Dict

import requests

from databutler.utils import pickleutils, caching
from scripts.mining.kaggle import utils


class NotebookSourceType(Enum):
    IPYTHON_NOTEBOOK = 0
    PYTHON_SOURCE_FILE = 1


def get_local_nb_storage_path() -> str:
    return os.path.join(utils.get_working_dir_for_mining(), "notebooks.pkl")


@caching.caching_function
def get_local_storage_reader() -> pickleutils.PickledMapReader:
    return pickleutils.PickledMapReader(get_local_nb_storage_path())


def fetch_notebook_data(username: str, kernel_slug: str) -> Dict:
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


def retrieve_notebook_data(username: str, kernel_slug: str) -> Dict:
    reader = get_local_storage_reader()
    if (username, kernel_slug) in reader:
        return reader[username, kernel_slug]["data"]
    else:
        return fetch_notebook_data(username, kernel_slug)


def get_docker_image(username: str, kernel_slug: str) -> str:
    """

    Args:
        username:
        kernel_slug:

    Returns:

    """
    nb_data = retrieve_notebook_data(username, kernel_slug)
    docker_image_digest = nb_data["kernelRun"]["runInfo"]["dockerImageDigest"]
    return f"gcr.io/kaggle-images/python@sha256:{docker_image_digest}"


def get_source_type(username: str, kernel_slug: str) -> NotebookSourceType:
    nb_data = retrieve_notebook_data(username, kernel_slug)
    source_type = nb_data["kernelRun"]["sourceType"]

    if source_type == "EDITOR_TYPE_NOTEBOOK":
        return NotebookSourceType.IPYTHON_NOTEBOOK
    else:
        raise ValueError(f"Could not recognize source type {source_type}")


def get_source(username: str, kernel_slug: str) -> str:
    nb_data = retrieve_notebook_data(username, kernel_slug)
    source = nb_data["kernelRun"]["commit"]["source"]

    return source
