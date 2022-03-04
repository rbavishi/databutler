import json
import os.path
from typing import List, Tuple, Optional

import fire

from databutler.utils.logging import logger
from scripts.mining.kaggle import utils
from scripts.mining.kaggle.execution.result import NotebookExecStatus
from scripts.mining.kaggle.notebooks.download import get_notebooks_using_meta_kaggle, download_notebooks
from scripts.mining.kaggle.notebooks import utils as nb_utils
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook
from scripts.mining.kaggle.utils import fire_command


@fire_command(name='download_notebooks_using_meta_kaggle', collection=__file__)
def download_notebooks_using_meta_kaggle():
    """
    Download all notebooks retrieved using the meta-kaggle dataset
    """
    meta_kaggle_notebooks: List[Tuple[str, str]] = get_notebooks_using_meta_kaggle()
    download_notebooks(meta_kaggle_notebooks)


@fire_command(name='download_notebook', collection=__file__)
def download_notebook(owner: str, slug: str):
    """
    Download a single notebook to local storage.

    Note that this is not necessary to run other commands such as view_notebook. Running this command
    helps eliminate the time spent in fetching it from the Kaggle servers.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)
    """
    nb_utils.download_notebook_data(owner, slug)
    print(f"Successfully downloaded notebook at {owner}/{slug}.")


@fire_command(name='view_notebook', collection=__file__)
def view_notebook(owner: str, slug: str):
    """
    View the raw Kaggle data for a notebook as a JSON.

    This is mostly intended for debugging. It is not necessary for a notebook to be downloaded.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)
    """
    nb_data = nb_utils.retrieve_notebook_data(owner, slug)
    print(json.dumps(nb_data, indent=2))


@fire_command(name='is_notebook_downloaded', collection=__file__)
def is_notebook_downloaded(owner: str, slug: str) -> bool:
    """
    Check if a notebook is downloaded locally.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)

    Returns:
        True if the notebook has been downloaded locally, and False otherwise.
    """
    return nb_utils.is_notebook_data_downloaded(owner, slug)


@fire_command(name='is_notebook_in_meta_kaggle', collection=__file__)
def is_notebook_in_meta_kaggle(owner: str, slug: str) -> bool:
    """
    Check if a specific notebook is included in the meta-kaggle dataset.

    This is mostly intended for debugging the meta-kaggle dataset. This function is expensive to run.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)

    Returns:
        True if the notebook is included in the Meta-Kaggle dataset, and False otherwise.
    """
    meta_kaggle_notebooks: List[Tuple[str, str]] = get_notebooks_using_meta_kaggle()
    return (owner, slug) in meta_kaggle_notebooks


@fire_command(name='download_notebook_data_sources', collection=__file__)
def download_notebook_data_sources(owner: str, slug: str, force: bool = False, verbose: bool = True):
    """
    Download all the data-sources for a notebook to local storage.

    This is mostly intended for debugging and manual inspection.
    Notebook runners, for example, will automatically download the data-sources if unavailable.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)
        force (bool): If True, the data-source is downloaded again if it's already available. Defaults to False.
        verbose (bool): If true, the downloading process is verbose. Defaults to True.
    """
    ds_list = KaggleNotebook(owner, slug).data_sources

    for ds in ds_list:
        if (not force) and ds.is_downloaded():
            print(f"Data-Source {ds.url} already downloaded.")
            continue
        elif ds.is_downloaded():
            print(f"Re-downloading {ds.url}...")
        else:
            print(f"Downloading {ds.url}...")

        try:
            ds.download(force=force, verbose=verbose)
        except Exception as e:
            print(f"[!] Failed to download data-source {ds.url}")
            logger.exception(e)
        else:
            print(f"Successfully downloaded data-source {ds.url} to {ds.local_storage_path}")


@fire_command(name='get_available_executors', collection=__file__)
def get_available_executors() -> List[str]:
    """
    Get all the available executors that can be provided to `run_notebook` and `run_notebooks`.
    """
    return [
        'mpl_seaborn_viz_miner',
    ]


@fire_command(name='run_notebook', collection=__file__)
def run_notebook(owner: str, slug: str, executor_name: str, output_dir_path: str, timeout: Optional[int] = None):
    """
    Run a notebook with the given executor.

    This is mostly intended for debugging. To run notebooks in bulk should be done, use `run_notebooks`.

    Args:
        owner (str): Username of the notebook owner.
        slug (str): Slug of the notebook (last segment of the notebook's URL)
        executor_name (str): Name of the executor to use.
            Use `cli.py get_available_executors` to get available executors.
        output_dir_path (str): Path to a directory on the local (host) filesystem where output, if any, of the
            executor will be saved.
        timeout (Optional[int]): If not None, the timeout (in seconds) to use for the notebook. Defaults to None.
    """
    if executor_name not in get_available_executors():
        raise ValueError(f"Executor {executor_name} not found. Executor must be one of {get_available_executors()}")

    if executor_name == "mpl_seaborn_viz_miner":
        from scripts.mining.kaggle.execution.mpl_seaborn_miner import MplSeabornVizMiner
        executor = MplSeabornVizMiner
    else:
        assert False

    notebook = KaggleNotebook(owner, slug)
    result = executor.run_notebook(notebook, output_dir_path=output_dir_path, timeout=timeout)

    if result.status == NotebookExecStatus.SUCCESS:
        print("Notebook execution succeeded.")
    elif result.status == NotebookExecStatus.ERROR:
        print("Notebook execution errored out.")
        print("Error message:")
        print(result.msg)
    elif result.status == NotebookExecStatus.TIMEOUT:
        print("Notebook execution timed out.")

    if os.path.exists(executor.get_stdout_log_path(output_dir_path)):
        print("STDOUT:")
        with open(executor.get_stdout_log_path(output_dir_path), "r") as f:
            print(f.read())

        print("----------------")

    if os.path.exists(executor.get_stderr_log_path(output_dir_path)):
        print("STDERR:")
        with open(executor.get_stderr_log_path(output_dir_path), "r") as f:
            print(f.read())

        print("----------------")


if __name__ == "__main__":
    fire.Fire(utils.get_fire_commands_for_collection(__file__))
