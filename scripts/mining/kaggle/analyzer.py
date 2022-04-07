from io import TextIOWrapper
import json
import os.path
from typing import List, Tuple, Optional, Callable, Dict, Iterator

import attrs
from pexpect import ExceptionPexpect
import fire
import tqdm

from databutler.utils.logging import logger
from scripts.mining.kaggle import utils
from scripts.mining.kaggle.execution.result import NotebookExecStatus
from scripts.mining.kaggle.notebooks import utils as nb_utils
from scripts.mining.kaggle.notebooks.download import get_notebooks_using_meta_kaggle, download_notebooks, \
    download_notebooks_gdrive
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook, KaggleNotebookSourceType
from scripts.mining.kaggle.utils import fire_command


@fire_command(name='get_nbs_importing_lib', collection=__file__)
def get_nb_links_importing_library(library_name: str, file_out: str, iter_num: int = None) -> None:
    """
    Save the links of files in the dataset that uses given library or
    any of its sub-libraries.
    This is a slow function, mainly intended for debugging, and should only be used sparingly.
    """
    file = open(file_out, 'w')
    nb_generator = iter_metakaggle_notebooks()
    # If iter_num is specified, we only iterate through the first n notebooks
    if iter_num is not None:
        for i in range(iter_num):
            nb = next(nb_generator)
            save_nb_if_library_imported(nb, library_name, file)

    # If iter_num is not specified, we iterate through all the notebooks
    else:
        for nb in nb_generator:
            save_nb_if_library_imported(nb, library_name, file)

    # Close the file at the end for memory efficiency
    file.close()


def save_nb_if_library_imported(nb: KaggleNotebook, library_name: str, file: TextIOWrapper) -> None:
    """
    Save the notebook link to the file if it uses the given library
    or any of its sub-libraries.
    """
    # Load the source code of the notebook
    try:
        nb_source = json.loads(nb.source_code)
    except Exception as e:
        logger.error(f'Failed to load source code of {nb.owner}/{nb.slug}')
        return

    # Get the cells of the notebook
    nb_cells = nb_source['cells']
    if nb_cells is None:
        logger.error(f'No cells in {nb.owner}/{nb.slug}, aborting library checker')
        return

    # Iterate through the cells to find any cell that imports the library
    # Save the notebook link if it is found
    try:
        if_library_imported = any([library_name in cell['source'] for cell in nb_cells])
        if if_library_imported:
            file.write(f'https://kaggle.com/{nb.owner}/{nb.slug}\n')
    except Exception as e:
        logger.error(f'Iterating through cells in {nb.owner}/{nb.slug} failed')


def iter_metakaggle_notebooks() -> Iterator[KaggleNotebook]:
    """
    Helper function that iterates through the notebooks on MetaKaggle dataset and yields them.
    """
    with nb_utils.get_local_nb_data_storage_reader() as reader:
        for (owner, slug) in tqdm.tqdm(reader.keys(), total=len(reader)):
            nb = KaggleNotebook.from_raw_data(owner, slug, reader[owner, slug])
            yield nb

if __name__ == "__main__":
    fire.Fire(utils.get_fire_commands_for_collection(__file__))

