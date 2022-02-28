import json
import logging
import os
from typing import Tuple, List, Set

import fire as fire
import pandas as pd
import tqdm

import utils
from databutler.utils import pickleutils
from databutler.utils.logging import logger

_OwnerUsername = str
_KernelSlug = str


def _get_download_path() -> str:
    return os.path.join(utils.get_working_dir_for_mining(), "notebooks.pkl")


def get_notebook_slugs() -> List[Tuple[_OwnerUsername, _KernelSlug]]:
    """
    Returns notebook slugs i.e. (owner-username, kernel-slug) tuples using the Meta-Kaggle dataset at
    https://www.kaggle.com/kaggle/meta-kaggle.

    Returns:
        (List[Tuple[_OwnerSlug, _KernelSlug]]): A list of tuples where the first element is the username of the
            owner and the second element is the kernel slug. The notebook's URL can then be constructed as
            https://www.kaggle.com/username/kernel-slug
    """
    #  Fetch all the useful data from the Meta-Kaggle dataset.
    kernels_csv_path = os.path.join(utils.get_working_dir_for_mining(), "Kernels.csv")
    users_csv_path = os.path.join(utils.get_working_dir_for_mining(), "Users.csv")
    kernel_versions_csv_path = os.path.join(utils.get_working_dir_for_mining(), "KernelVersions.csv")

    for path in [kernels_csv_path, users_csv_path, kernel_versions_csv_path]:
        if not os.path.exists(path):
            #  Since the process of downloading these CSVs needs authentication, we can't download it automatically.
            raise FileNotFoundError(
                f"Please download {os.path.basename(path)} from https://www.kaggle.com/kaggle/meta-kaggle "
                f"and place it at {path}"
            )

    #  First level of join helps us match up the kernel slug with the owner slug.
    kernels = pd.read_csv(kernels_csv_path, usecols=['AuthorUserId', 'CurrentKernelVersionId', 'CurrentUrlSlug'])
    users = pd.read_csv(users_csv_path, usecols=['Id', 'UserName'])

    m_kernel_user = kernels.merge(users, left_on=['AuthorUserId'], right_on=['Id'])[[
        'CurrentKernelVersionId', 'UserName', 'CurrentUrlSlug'
    ]]

    del kernels
    del users

    #  We also only want the Python ones, so we use the language field to filter those out.
    kernel_versions = pd.read_csv(kernel_versions_csv_path, usecols=['Id', 'ScriptLanguageId'])
    m_kernel_user_lang = m_kernel_user.merge(kernel_versions, left_on=['CurrentKernelVersionId'], right_on=['Id'])[[
        'UserName', 'CurrentUrlSlug', 'ScriptLanguageId'
    ]]

    python_lang_ids = {2, 8, 9, 14}  # https://www.kaggle.com/kaggle/meta-kaggle?select=KernelLanguages.csv
    m_kernel_user_lang = m_kernel_user_lang[m_kernel_user_lang.ScriptLanguageId.isin(python_lang_ids)]

    #  All set now
    result = set()
    for row in m_kernel_user_lang.itertuples():
        result.add((row.UserName, row.CurrentUrlSlug))

    return list(result)


def download_notebooks() -> None:
    """
    Downloads Kaggle notebooks using the slugs obtained from Meta-Kaggle analysis.

    NOTE: This is not multiprocess/multithread safe if used in conjunction with other storage updating functions.
    """
    slugs: List[Tuple[_OwnerUsername, _KernelSlug]] = list(get_notebook_slugs())[:10]

    download_path = _get_download_path()
    existing_keys: Set[Tuple[_OwnerUsername, _KernelSlug]] = set()
    if os.path.exists(download_path):
        with pickleutils.PickledMapReader(download_path) as reader:
            existing_keys.update(reader.keys())

    todo: List[Tuple[_OwnerUsername, _KernelSlug]] = [i for i in slugs if i not in existing_keys]

    logger.info(f"Found {len(slugs)} notebooks eligible for downloading.")
    logger.info(f"Already downloaded {len(existing_keys)} notebooks. Downloading {len(todo)} notebooks")

    with pickleutils.PickledMapWriter(download_path, overwrite_existing=False) as writer, \
            tqdm.tqdm(todo, desc="Downloading Notebooks", dynamic_ncols=True) as pbar:

        n_succ = n_fail = 0
        for owner_username, kernel_slug in pbar:
            try:
                nb_data = {
                    "owner_slug": owner_username,
                    "kernel_slug": kernel_slug,
                    "data": utils.get_notebook_data(owner_username, kernel_slug)
                }

                writer[owner_username, kernel_slug] = nb_data
                #  Flushing in order to be safe against interruptions.
                writer.flush()

                n_succ += 1

            except KeyboardInterrupt:
                break

            except:
                n_fail += 1

            pbar.set_postfix(success=n_succ, fail=n_fail)


def download_notebook(owner_username: str, kernel_slug: str) -> None:
    """
    A convenience method for downloading a single notebook, if it exists. This will however update the master cache
    for notebooks.

    NOTE: This is not multiprocess/multithread safe if used in conjunction with other storage updating functions.

    Args:
        owner_username: A string corresponding to the username of the owner.
        kernel_slug: A string corresponding to the kernel slug.
    """
    download_path = _get_download_path()
    url = f"https://kaggle.com/{owner_username}/{kernel_slug}"
    if os.path.exists(download_path):
        with pickleutils.PickledMapReader(download_path) as reader:
            if (owner_username, kernel_slug) in reader:
                print(f"Notebook at {url} already downloaded.")
                return

    try:
        nb_data = {
            "owner_slug": owner_username,
            "kernel_slug": kernel_slug,
            "data": utils.get_notebook_data(owner_username, kernel_slug)
        }

    except Exception as e:
        logging.exception(f"Failed to download notebook at {url}")

    else:
        with pickleutils.PickledMapWriter(download_path, overwrite_existing=False) as writer:
            writer[owner_username, kernel_slug] = nb_data

        print(f"Successfully downloaded notebook at {url}.")


def view_downloaded_notebook(owner_username: str, kernel_slug: str) -> None:
    download_path = _get_download_path()
    url = f"https://kaggle.com/{owner_username}/{kernel_slug}"

    if not os.path.exists(download_path):
        print(f"No notebooks downloaded so far.")
        return

    with pickleutils.PickledMapReader(download_path) as reader:
        if (owner_username, kernel_slug) not in reader:
            print(f"Notebook at {url} has not been downloaded yet. "
                  f"Try running `python downloader.py download_notebook {owner_username} {kernel_slug}`.")

            return

        nb_data = reader[owner_username, kernel_slug]["data"]
        print(json.dumps(nb_data, indent=2))


def generate_report_for_downloaded_notebooks():
    pass


if __name__ == "__main__":
    fire.Fire()
