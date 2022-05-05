import json
import os
import re
import traceback
from contextlib import contextmanager
from typing import Dict

import requests

from databutler.utils import pickleutils, caching
from databutler.mining.kaggle import utils
from databutler.mining.kaggle.exceptions import NotebookFetchError
from databutler.mining.kaggle.notebooks.scraping import convert_kaggle_html_to_ipynb


@caching.caching_function
def get_kaggle_api():
    #  We delay the import as long as possible to avoid crashes when the Kaggle credentials are not available locally.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_local_nb_data_storage_path() -> str:
    """
    Returns the path where the notebooks will be stored. This corresponds to a single PickledMap file.
    """
    return os.path.join(utils.get_working_dir_for_mining(), "notebooks.pkl")


def get_local_datasources_storage_path() -> str:
    """
    Returns the path where the datasources will be stored.
    """
    path = os.path.join(utils.get_working_dir_for_mining(), "datasources")
    os.makedirs(path, exist_ok=True)
    return path


@contextmanager
def get_local_nb_data_storage_reader() -> pickleutils.PickledMapReader:
    """
    Returns a PickledMap reader for the local notebook storage.
    """
    path = get_local_nb_data_storage_path()
    if not os.path.exists(path):
        #  Just create an empty map.
        with pickleutils.PickledMapWriter(path):
            pass

    with pickleutils.PickledMapReader(path) as reader:
        yield reader


@contextmanager
def get_local_nb_data_storage_writer() -> pickleutils.PickledMapWriter:
    """
    Returns a PickledMap writer for the local notebook storage.
    """
    path = get_local_nb_data_storage_path()
    with pickleutils.PickledMapWriter(path, overwrite_existing=False) as writer:
        yield writer


def fetch_notebook_data(owner: str, slug: str) -> Dict:
    """
    Fetches all the metadata, including sources, for a notebook given the username of the owner and the kernel slug.

    This uses requests to fetch the notebook using URL and uses some regex to get the notebook JSON out.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)

    Raises:
        NotebookFetchError: If data could not be downloaded for a notebook.

    Returns:
        (Dict): A dictionary containing the notebook data.
    """
    resp = requests.get(f"https://www.kaggle.com/{owner}/{slug}")

    if resp.status_code != 200:
        raise NotebookFetchError(f"GET request returned status code {resp.status_code}")

    text = resp.text

    #  NOTE: Although this is not too likely to change, this method will break if Kaggle decides to change their
    #  website structure.
    try:
        d = next(i for i in re.compile(r"Kaggle.State.push\(({.*})\)").findall(text) if "runInfo" in i)
    except StopIteration:
        raise NotebookFetchError(f"Regex matching failure for {owner}/{slug}")

    d = json.loads(d)

    if "kernelBlob" not in d:
        #  A recent Kaggle change removed this field from the source.
        #  For notebooks, we can try to use a kaggleusercontent.com link to get the notebook html rendering,
        #  and extract the script from that.
        kernel_run = d.get("kernelRun", {})
        source_type = kernel_run.get("sourceType", None)
        output_url = kernel_run.get("renderedOutputUrl", None)
        if source_type == "EDITOR_TYPE_NOTEBOOK" and "kaggleusercontent.com" in output_url:
            #  Try scraping directly
            try:
                resp = requests.get(output_url)

                if resp.status_code != 200:
                    raise RuntimeError

                text = resp.text
                d["kernelBlob"] = {
                    "source": json.dumps(convert_kaggle_html_to_ipynb(text))
                }

            except:
                pass

            else:
                return d

        #  Rely on the Kaggle API
        api = get_kaggle_api()
        try:
            response = api.process_response(api.kernel_pull_with_http_info(owner, slug))
        except Exception as e:
            raise NotebookFetchError(f"Failed to fetch kernel sources")

        d['kernelBlob'] = response['blob']

    return d


def download_notebook_data(owner: str, slug: str):
    """
    Saves notebook data to the local storage.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)
    """
    with get_local_nb_data_storage_writer() as writer:
        if (owner, slug) not in writer:
            writer[owner, slug] = fetch_notebook_data(owner, slug)


def is_notebook_data_downloaded(owner: str, slug: str) -> bool:
    """
    Check if a notebook has been downloaded to local storage.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)

    Returns:
        True if notebook data has been downloaded, False otherwise.
    """
    with get_local_nb_data_storage_reader() as reader:
        return (owner, slug) in reader


def retrieve_notebook_data(owner: str, slug: str) -> Dict:
    """
    Equivalent in functionality to fetch_notebook_data, with the only difference being that this routine
    consults the local storage first.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)

    Returns:
        (Dict): A dictionary containing the notebook data.
    """
    with get_local_nb_data_storage_reader() as reader:
        if (owner, slug) in reader:
            return reader[owner, slug]

    return fetch_notebook_data(owner, slug)
