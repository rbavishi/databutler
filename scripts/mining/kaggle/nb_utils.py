import glob
import json
import os
import re
from enum import Enum
from typing import Dict, List

import attrs
import requests

from databutler.utils import pickleutils, caching
from scripts.mining.kaggle import utils


class KaggleSourceType(Enum):
    """
    Source type of a Kaggle notebook.
    """
    IPYTHON_NOTEBOOK = 0
    PYTHON_SOURCE_FILE = 1


class KaggleDataSourceType(Enum):
    """
    Type of the data-source associated with a notebook.

    Competition data is data such as that of titanic or house-prices.
    Datasets are those that are uploaded by individuals.
    Kernel outputs are similar to datasets, but are associated with a specific kernel version.
    """
    COMPETITION = 0
    DATASET = 1
    KERNEL_OUTPUT = 2
    UNKNOWN = 3


@attrs.define
class KaggleDataSource:
    """
    Holds information about a data-source.
    """
    #  URL relative to kaggle.com
    url: str
    #  Mount location inside /kaggle/input.
    mount_slug: str
    #  Competition, Dataset or Kernel data.
    src_type: KaggleDataSourceType
    #  Where it will be stored locally, if downloaded.
    local_storage_path: str


@caching.caching_function
def get_kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_local_nb_storage_path() -> str:
    """
    Returns the path where the notebooks will be stored. This corresponds to a single PickledMap file.
    """
    return os.path.join(utils.get_working_dir_for_mining(), "notebooks.pkl")


def get_local_datasources_storage_path() -> str:
    """
    Returns the path where the notebooks will be stored. This corresponds to a single PickledMap file.
    """
    path = os.path.join(utils.get_working_dir_for_mining(), "datasources")
    os.makedirs(path, exist_ok=True)
    return path


@caching.caching_function
def get_local_storage_reader() -> pickleutils.PickledMapReader:
    """
    Returns a PickledMap reader for the local notebook storage.
    """
    path = get_local_nb_storage_path()
    if not os.path.exists(path):
        #  Just create an empty map.
        with pickleutils.PickledMapWriter(path):
            pass

    return pickleutils.PickledMapReader(get_local_nb_storage_path())


def fetch_notebook_data(username: str, kernel_slug: str) -> Dict:
    """
    Fetches all the metadata, including sources, for a notebook given the username of the owner and the kernel slug.

    This uses requests to fetch the notebook using URL and uses some regex to get the notebook JSON out.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

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
    """
    Returns all the metadata, including sources, for a notebook given the username of the owner and the kernel slug.

    This is like `fetch_notebook_data` with the primary difference being that this routine consults the local notebook
    storage first.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        (Dict): A dictionary containing the notebook data.
    """
    reader = get_local_storage_reader()
    if (username, kernel_slug) in reader:
        return reader[username, kernel_slug]["data"]
    else:
        return fetch_notebook_data(username, kernel_slug)


def get_docker_image_digest(username: str, kernel_slug: str) -> str:
    """
    Retrieves the docker image digest for the latest run (as per Meta-Kaggle) of the given notebook.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        (str): A string corresponding to the digest.
    """
    nb_data = retrieve_notebook_data(username, kernel_slug)
    return nb_data["kernelRun"]["runInfo"]["dockerImageDigest"]


def get_docker_image_url(username: str, kernel_slug: str) -> str:
    """
    Returns the full docker image URL (hosted on gcr.io) for the latest run (as per Meta-Kaggle) of the given notebook.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        (str): A string corresponding to the URL.
    """
    docker_image_digest = get_docker_image_digest(username, kernel_slug)
    return f"gcr.io/kaggle-images/python@sha256:{docker_image_digest}"


def get_source_type(username: str, kernel_slug: str) -> KaggleSourceType:
    """
    Maps the source type of the notebook to the internal KaggleSourceType.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Raises:
        ValueError: If the source type is not recognized.

    Returns:
        (KaggleSourceType): An enum member corresponding to the source type.
    """
    nb_data = retrieve_notebook_data(username, kernel_slug)
    source_type = nb_data["kernelRun"]["sourceType"]

    if source_type == "EDITOR_TYPE_NOTEBOOK":
        return KaggleSourceType.IPYTHON_NOTEBOOK
    else:
        raise ValueError(f"Could not recognize source type {source_type}")


def get_source(username: str, kernel_slug: str) -> str:
    """
    Returns the source of a notebook.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        (str): A string corresponding to the source of the notebook.

    """
    nb_data = retrieve_notebook_data(username, kernel_slug)
    source = nb_data["kernelRun"]["commit"]["source"]

    return source


def get_data_sources(username: str, kernel_slug: str) -> List[KaggleDataSource]:
    """
    Fetches all the datasources defined for a notebook.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        (List[KaggleDataSource]): A list of data-sources.
    """
    nb_data = retrieve_notebook_data(username, kernel_slug)
    result: List[KaggleDataSource] = []
    ds_root = get_local_datasources_storage_path()

    for ds in nb_data["renderableDataSources"]:
        try:
            url = ds["dataSourceUrl"]
            mount_slug = ds["reference"]["mountSlug"]
            native_type = ds["reference"].get("sourceType", "")

            #  Determine the type of data-source using certain heuristics.
            if url.startswith("/c/") and len(url.split('/')) == 3:
                #  If it starts with "/c/", it is bound to be a competition data-source.
                src_type = KaggleDataSourceType.COMPETITION
                local_storage_path = os.path.join(ds_root, "c", url.split("/")[-1])

            elif url.startswith("/") and len(url.split('/')) == 3 and "DATASET_VERSION" in native_type:
                #  We use the JSON extracted from the notebook's webpage.
                src_type = KaggleDataSourceType.DATASET
                local_storage_path = os.path.join(ds_root, "d", *url.split("/")[1:])

            elif url.startswith("/") and len(url.split('/')) == 3 and "KERNEL_VERSION" in native_type:
                #  We use the JSON extracted from the notebook's webpage.
                src_type = KaggleDataSourceType.KERNEL_OUTPUT
                local_storage_path = os.path.join(ds_root, "k", *url.split("/")[1:])

            else:
                src_type = KaggleDataSourceType.UNKNOWN
                local_storage_path = ""

            result.append(KaggleDataSource(
                url=url,
                mount_slug=mount_slug,
                src_type=src_type,
                local_storage_path=local_storage_path,
            ))

        except KeyError:
            pass

    return result


def are_data_sources_available(username: str, kernel_slug: str) -> bool:
    """
    Checks if all the data-sources for a notebook are available locally.

    Args:
        username (str): Username of the notebook owner.
        kernel_slug (str): Slug of the kernel used in the notebook's URL.

    Returns:
        True if all data-sources are available, else False.
    """
    ds_list: List[KaggleDataSource] = get_data_sources(username, kernel_slug)
    return all(os.path.exists(ds.local_storage_path) for ds in ds_list)


def download_data_source(ds: KaggleDataSource, force: bool = False, quiet: bool = True):
    """
    Downloads a particular data-source to local storage.

    Args:
        ds (KaggleDataSource): A data-source.
        force (bool): Force a re-download if the data-source already exists. Defaults to False.
        quiet (bool): If false, progress is displayed. Defaults to True.
    """
    api = get_kaggle_api()

    if ds.src_type == KaggleDataSourceType.COMPETITION:
        competition_slug: str = ds.url.split("/")[-1]
        path = ds.local_storage_path

        if (not os.path.exists(path)) or force:
            os.makedirs(path, exist_ok=True)
            api.competition_download_files(competition=competition_slug, path=path, force=force, quiet=quiet)

            #  If it is a zip, unzip it.
            outfile = os.path.join(path, competition_slug + '.zip')
            if os.path.exists(outfile):
                utils.unzip(outfile, path, remove_zip=True)

            #  Unzip any zip inside
            for p in glob.glob(os.path.join(path, "*.zip")):
                utils.unzip(p, os.path.dirname(p), remove_zip=False)

    elif ds.src_type == KaggleDataSourceType.DATASET:
        owner, dataset_slug = ds.url.split('/')[1:]

        path = ds.local_storage_path
        os.makedirs(path, exist_ok=True)

        api.dataset_download_files(dataset=f"{owner}/{dataset_slug}", path=path, force=force, quiet=quiet, unzip=True)

    elif ds.src_type == KaggleDataSourceType.KERNEL_OUTPUT:
        owner, dataset_slug = ds.url.split('/')[1:]

        path = ds.local_storage_path
        os.makedirs(path, exist_ok=True)
        api.kernels_pull(kernel=f"{owner}/{dataset_slug}", path=path, quiet=quiet, metadata=False)
        api.kernels_output(kernel=f"{owner}/{dataset_slug}", path=path, force=force, quiet=quiet)

    else:
        raise NotImplementedError(f"Unrecognized data-source {ds.url}")
