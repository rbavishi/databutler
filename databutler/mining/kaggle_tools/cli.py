import json
import os.path
from typing import List, Tuple, Optional, Callable, Dict, Iterator

import attrs
import fire
import tqdm

from databutler.utils.logging import logger
from databutler.mining.kaggle_tools import utils
from databutler.mining.kaggle_tools.execution.result import NotebookExecStatus
from databutler.mining.kaggle_tools.notebooks import utils as nb_utils
from databutler.mining.kaggle_tools.notebooks.download import (
    get_notebooks_using_meta_kaggle,
    download_notebooks,
    download_notebooks_gdrive,
)
from databutler.mining.kaggle_tools.notebooks.notebook import (
    KaggleNotebook,
    KaggleNotebookSourceType,
)
from databutler.mining.kaggle_tools.utils import fire_command


@fire_command(name="MetaKaggleNotebooks", collection=__file__)
@attrs.define(eq=False, repr=False)
class MetaKaggleNotebooks:
    """
    Make a selection of notebooks using the meta-kaggle dump by filtering across criteria.
    """

    _filters: List[Callable[[KaggleNotebook], bool]] = attrs.field(
        init=False, factory=list
    )

    def set_filter(
        self,
        competition_only: bool = True,
        pure_competition_only: bool = None,
        competition: Optional[str] = None,
        successful_execution: Optional[bool] = True,
        author: Optional[str] = None,
        max_runtime: Optional[int] = None,
        is_gpu_accelerated: Optional[bool] = None,
        has_docker_image_available: Optional[bool] = None,
        is_notebook: Optional[bool] = None,
        libraries_used: Optional[List[str]] = None,
        libraries_not_used: Optional[List[str]] = None,
    ):
        """

        Args:
            competition_only:
            pure_competition_only:
            competition:
            successful_execution:
            author:
            max_runtime:
            is_gpu_accelerated:
            has_docker_image_available:
            is_notebook:
            libraries_used:
            libraries_not_used:
        """
        self._filters.clear()

        if competition_only:
            self._filters.append(lambda nb: nb.associated_competition is not None)
        if pure_competition_only:
            self._filters.append(lambda nb: nb.is_pure_competition_notebook())
        if competition is not None:
            self._filters.append(lambda nb: nb.associated_competition == competition)
        if successful_execution is not None:
            self._filters.append(
                lambda nb: nb.was_execution_successful() == successful_execution
            )
        if author is not None:
            self._filters.append(lambda nb: nb.owner == author)
        if max_runtime is not None:
            self._filters.append(
                lambda nb: nb.runtime is not None and nb.runtime <= max_runtime
            )
        if is_gpu_accelerated is not None:
            self._filters.append(
                lambda nb: nb.is_gpu_accelerated() == is_gpu_accelerated
            )
        if has_docker_image_available:
            #  TODO: Need a way to check the registry
            pass
        if is_notebook is not None:
            self._filters.append(
                lambda nb: nb.source_type == KaggleNotebookSourceType.IPYTHON_NOTEBOOK
            )
        if libraries_used is not None:
            set_libraries_used = set(libraries_used)
            self._filters.append(
                lambda nb: set_libraries_used.issubset(nb.imported_packages)
            )
        if libraries_used is not None:
            set_libraries_not_used = set(libraries_not_used)
            self._filters.append(
                lambda nb: set_libraries_not_used.isdisjoint(nb.imported_packages)
            )

        #  Allow chaining of commands
        return self

    def _iter_selection(self) -> Iterator[KaggleNotebook]:
        with nb_utils.get_local_nb_data_storage_reader() as reader:
            for (owner, slug) in tqdm.tqdm(reader.keys(), total=len(reader)):
                nb = KaggleNotebook.from_raw_data(owner, slug, reader[owner, slug])
                if all(fn(nb) for fn in self._filters):
                    yield nb

    def generate_report(self) -> None:
        total = 0
        competitions = set()
        for nb in self._iter_selection():
            total += 1
            competitions.add(nb.associated_competition)

        print(f"Found {total} notebooks meeting criteria")
        print(f"Found {len(competitions)} competitions in total.")

    def dump_json(self, path: str):
        pass

    def load_json(self, path: str):
        pass


@fire_command(name="download_notebooks_using_meta_kaggle", collection=__file__)
def download_notebooks_using_meta_kaggle(num_processes: int = 1):
    """
    Download all notebooks retrieved using the meta-kaggle dataset
    """
    meta_kaggle_notebooks: List[Tuple[str, str]] = get_notebooks_using_meta_kaggle()
    download_notebooks(meta_kaggle_notebooks, num_processes=num_processes)


@fire_command(name="download_notebooks_from_drive", collection=__file__)
def download_notebooks_from_drive():
    download_notebooks_gdrive()


@fire_command(name="download_notebook", collection=__file__)
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


@fire_command(name="view_notebook", collection=__file__)
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
    from databutler.mining.kaggle_tools.notebooks.notebook import KaggleNotebook

    nb = KaggleNotebook(owner, slug)
    print(nb.associated_competition)
    print(nb.is_gpu_accelerated())
    print(nb.is_pure_competition_notebook())
    print(nb.runtime)
    print(nb.was_execution_successful())
    print(nb.imported_packages)


@fire_command(name="is_notebook_downloaded", collection=__file__)
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


@fire_command(name="is_notebook_in_meta_kaggle", collection=__file__)
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


@fire_command(name="download_notebook_data_sources", collection=__file__)
def download_notebook_data_sources(
    owner: str, slug: str, force: bool = False, verbose: bool = True
):
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
            print(
                f"Successfully downloaded data-source {ds.url} to {ds.local_storage_path}"
            )


@fire_command(name="get_available_executors", collection=__file__)
def get_available_executors() -> List[str]:
    """
    Get all the available executors that can be provided to `run_notebook` and `run_notebooks`.
    """
    return [
        "simple_executor",
        "pandas_miner",
        "plotly_miner",
        "mpl_seaborn_viz_miner",
    ]


@fire_command(name="run_notebook", collection=__file__)
def run_notebook(
    owner: str,
    slug: str,
    executor_name: str,
    output_dir_path: str,
    docker_image_url: Optional[str] = None,
    timeout: Optional[int] = None,
    **executor_kwargs,
):
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
        docker_image_url (Optional[str]): If not None, the docker image corresponding to the URL is used instead
            of the one associated with the notebook.
        timeout (Optional[int]): If not None, the timeout (in seconds) to use for the notebook. Defaults to None.
        **executor_kwargs: Additional keyword arguments specific to the executor.
    """
    if executor_name not in get_available_executors():
        raise ValueError(
            f"Executor {executor_name} not found. Executor must be one of {get_available_executors()}"
        )

    if executor_name == "simple_executor":
        from databutler.mining.kaggle_tools.execution.simple_executor import (
            SimpleExecutor,
        )

        executor = SimpleExecutor

    elif executor_name == "pandas_miner":
        from databutler.mining.kaggle_tools.dynamic_analysis.pandas_mining.miner import (
            PandasMiner,
        )

        executor = PandasMiner

    elif executor_name == "plotly_miner":
        from databutler.mining.kaggle_tools.dynamic_analysis.plotly_mining.miner import (
            PlotlyMiner,
        )

        executor = PlotlyMiner

    elif executor_name == "mpl_seaborn_viz_miner":
        from databutler.mining.kaggle_tools.dynamic_analysis.mpl_seaborn_mining.miner import (
            MplSeabornVizMiner,
        )

        executor = MplSeabornVizMiner

    else:
        assert False

    notebook = KaggleNotebook(owner, slug)
    result = executor.run_notebook(
        notebook,
        output_dir_path=output_dir_path,
        docker_image_url=docker_image_url,
        timeout=timeout,
        **executor_kwargs,
    )

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
