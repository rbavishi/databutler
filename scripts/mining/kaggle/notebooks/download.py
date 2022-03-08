import os
from typing import Tuple, List, Set

import fire as fire
import pandas as pd
import tqdm

from databutler.utils import multiprocess, gdriveutils
from databutler.utils.logging import logger
from scripts.mining.kaggle import utils
from scripts.mining.kaggle.notebooks import utils as nb_utils

_Owner = str
_Slug = str


def get_notebooks_using_meta_kaggle() -> List[Tuple[_Owner, _Slug]]:
    """
    Returns notebook slugs i.e. (owner-username, kernel-slug) tuples using the Meta-Kaggle dataset at
    https://www.kaggle.com/kaggle/meta-kaggle.

    Returns:
        (List[Tuple[_OwnerSlug, _Slug]]): A list of tuples where the first element is the username of the
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


def _mp_helper_fetch_notebook_data(arg):
    owner, slug = arg
    return owner, slug, nb_utils.fetch_notebook_data(owner, slug)


def download_notebooks_gdrive() -> None:
    folder_id: str = "1xFcDRqY2QNLstn00-gxOa24Z64M788xQ"
    gdriveutils.download_folder(folder_id, os.path.dirname(nb_utils.get_local_datasources_storage_path()),
                                only_contents=True)


def download_notebooks(notebooks: List[Tuple[_Owner, _Slug]], num_processes: int = 1) -> None:
    """
    Downloads all provided notebooks as tuples of owner and slugs.

    NOTE: This is not multiprocess/multithread safe if used in conjunction with other storage updating functions.
    """

    existing_keys: Set[Tuple[_Owner, _Slug]] = set()
    with nb_utils.get_local_nb_data_storage_reader() as reader:
        existing_keys.update(reader.keys())

    todo: List[Tuple[_Owner, _Slug]] = [i for i in notebooks if i not in existing_keys]

    logger.info(f"Already downloaded {len(existing_keys)} notebooks. Downloading {len(todo)} notebooks")

    with nb_utils.get_local_nb_data_storage_writer() as writer:
        if num_processes == 1:
            with tqdm.tqdm(todo, desc="Downloading Notebooks", dynamic_ncols=True) as pbar:
                n_succ = n_fail = 0
                for owner, slug in pbar:
                    try:
                        nb_data = nb_utils.fetch_notebook_data(owner, slug)
                        writer[owner, slug] = nb_data
                        #  Flushing in order to be safe against interruptions.
                        writer.flush()

                        n_succ += 1

                    except KeyboardInterrupt:
                        break

                    except Exception as e:
                        logger.warning(f"Failed for {owner}/{slug}")
                        logger.exception(e)
                        n_fail += 1

                    pbar.set_postfix(success=n_succ, fail=n_fail)

        else:
            iterator = multiprocess.run_tasks_in_parallel_iter(_mp_helper_fetch_notebook_data,
                                                               todo,
                                                               use_progress_bar=True,
                                                               use_spawn=True,
                                                               num_workers=num_processes,
                                                               progress_bar_desc="Downloading Notebooks",
                                                               timeout_per_task=30)
            for task_result in iterator:
                if task_result.is_success():
                    try:
                        owner, slug, nb_data = task_result.result
                        writer[owner, slug] = nb_data
                        #  Flushing in order to be safe against interruptions.
                        writer.flush()

                    except KeyboardInterrupt:
                        break

                    except Exception as e:
                        logger.exception(e)

                elif task_result.is_exception():
                    logger.exception(task_result.exception_tb)


if __name__ == "__main__":
    fire.Fire()
