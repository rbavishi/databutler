from io import TextIOWrapper
import json
import os.path
from typing import List, Tuple, Optional, Callable, Dict, Iterator

import attrs
import fire
import tqdm
import json
import random
import webbrowser

from databutler.utils.logging import logger
from scripts.mining.kaggle import utils
from scripts.mining.kaggle.execution.result import NotebookExecStatus
from scripts.mining.kaggle.notebooks import utils as nb_utils
from scripts.mining.kaggle.notebooks.download import get_notebooks_using_meta_kaggle, download_notebooks, \
    download_notebooks_gdrive
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook, KaggleNotebookSourceType
from scripts.mining.kaggle.utils import fire_command

# The analyzer is a human-interpretable way to analyze the massive dataset that is MetaKaggle.
# We provide the following features:
# 1. saving links of files that have certain imports
# 2. randomly opening links once these files have been saved
# 3. providing statistics with relation to the entire metakaggle dataset
# 4. providing statistics within the library subset, giving search functionality for strings within this subset

METAKAGGLE_SIZE = 479454

@fire_command(name='get_nbs_importing_lib', collection=__file__)
def get_nb_links_importing_library(library_name: str, outdir_path: str, iter_num: int = None) -> None:
    """
    Save the links of files in the dataset that uses given library or
    any of its sub-libraries.
    This is a slow function, mainly intended for debugging, and should only be used sparingly.
    """

    def save_nb_if_library_imported(nb: KaggleNotebook, library_name: str, links: List) -> None:
        """
        Save the notebook link to the file if it uses the given library
        or any of its sub-libraries. Returns true if contains library, false otherwise.
        """
        # Treat the source code as one big string and look for an instance of library name.
        # This is a heuristic — it might have some false positives, but will never have a false negative.
        if library_name in nb.source_code:
            links.append(f'https://kaggle.com/{nb.owner}/{nb.slug}\n')
            return True
        return False

    if not os.path.exists(outdir_path):
        logger.info(f'Output directory {outdir_path} does not exist, creating')
        os.makedirs(outdir_path)
    file = open(f'{outdir_path}/{library_name}.json', 'w')

    lib_data = {
        'ran_all': iter_num is None,
        'found_count': 0,
        'links': []
    }

    count = 0
    nb_generator = iter_metakaggle_notebooks()
    # If iter_num is specified, we only iterate through the first n notebooks
    if iter_num is not None:
        for i in range(iter_num):
            nb = next(nb_generator)
            count += save_nb_if_library_imported(nb, library_name, lib_data['links'])

    # If iter_num is not specified, we iterate through all the notebooks
    else:
        for nb in nb_generator:
            count += save_nb_if_library_imported(nb, library_name, lib_data['links'])

    lib_data['found_count'] = count

    # Writing data to file
    json.dump(lib_data, file)
    # Close the file at the end for memory efficiency
    file.close()

    # If we've run all notebooks, display the data about our run automatically
    if lib_data['ran_all']:
        lib_set_stats(library_name, outdir_path)

@fire_command(name='random_open', collection=__file__)
def random_open_from_lib_list(lib_name: str, outdir_path: str, size: int = 10) -> None:
    """
    Accepts the name of a library and the output directory path, and opens size number of random files
    from the list.
    """
    if not os.path.exists(f'{outdir_path}/{lib_name}.json'):
        logger.error(f'Data file for library {lib_name} does not exist in {outdir_path}. Please run "get_nbs_importing_lib" first')
        return

    file = open(f'{outdir_path}/{lib_name}.json', 'r')
    try:
        lib_data = json.load(file)
    except Exception as e:
        logger.error(f'Data loading from {outdir_path}/{lib_name}.json failed, even though file exists. Data might be corrupted.')

    if size > lib_data['found_count']:
        logger.info(f'Sample size provided bigger than saved files — opening all')
        size = lib_data['found_count']

    samples = random.sample(lib_data['links'], size)
    for sample_link in samples:
        webbrowser.open(sample_link)

    logger.info(f'Opened {size} random files from list of files in browser')

@fire_command(name='display_lib_stats', collection=__file__)
def lib_set_stats(lib_name: str, outdir_path: str) -> None:
    """
    Accepts the name of the library being analyzed, and the path where its data is saved,
    returns data about the files with relation to the metakaggle dataset.
    """
    #TODO: Return more information, like how much competitions, etc

    if not os.path.exists(f'{outdir_path}/{lib_name}.json'):
        logger.error(f'Data file for library {lib_name} does not exist in {outdir_path}. Please run "get_nbs_importing_lib" first')
        return

    file = open(f'{outdir_path}/{lib_name}.json', 'r')
    try:
        lib_data = json.load(file)
    except Exception as e:
        logger.error(f'Data loading from {outdir_path}/{lib_name}.json failed, even though file exists. Data might be corrupted.')

    count = lib_data['found_count']
    if not lib_data['ran_all']:
        logger.warning(f"Haven't run library checker on all metakaggle notebooks — this stat is incorrect.")
    logger.info(f"Found {count} notebooks containing {lib_name} — that's {count / METAKAGGLE_SIZE * 100}% of all notebooks mined. (May be an overestimate)")


@fire_command(name='search_str_instances', collection=__file__)
def search_str_instances(search_str, lib_name: str, outdir_path: str, cutoff = 1000):
    """
    Searches for a particular string's existence within the list of notebooks saved
    containing a particular library.
    """
    def get_nb_retrieval_data(link):
        """
        Returns the owner and slug of the link.
        """
        split_url = link.replace('\n', '').split('/')
        return split_url[3], split_url[4]

    def _str_instance_is_in_nb(search_str, nb_data):
        """
        Returns True if the search string is in notebook data, false otherwise.
        """
        nb_source = nb_data['kernelBlob']['source']
        try:
            source = json.loads(nb_source)
            cells = source['cells']
            does_string_exist = any([search_str in line['source'] for line in cells])
            return does_string_exist
        except Exception as e:
            logger.error(f'Could not read notebook cells, abandoning notebook')
            return False

    if not os.path.exists(f'{outdir_path}/{lib_name}.json'):
        logger.error(f'Data file for library {lib_name} does not exist in {outdir_path}. Please run "get_nbs_importing_lib" first')
        return

    file = open(f'{outdir_path}/{lib_name}.json', 'r')
    try:
        lib_data = json.load(file)
    except Exception as e:
        logger.error(f'Data loading from {outdir_path}/{lib_name}.json failed, even though file exists. Data might be corrupted.')

    string_count = 0
    total_nb = lib_data['found_count']
    total_instances = cutoff if cutoff < total_nb else total_nb
    for i, line in tqdm.tqdm(enumerate(lib_data['links']), total=total_instances):
        if cutoff == 0: break
        cutoff -= 1
        owner, slug = get_nb_retrieval_data(line)
        nb_data = nb_utils.retrieve_notebook_data(owner, slug)
        if _str_instance_is_in_nb(search_str, nb_data):
            string_count += 1

    logger.info(f'\nAnalysing {total_instances}, {string_count} contained "{search_str}" | {string_count / total_instances * 100}%')


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

