import json
import os
import pickle
import re
from typing import Set, Tuple, Dict, List

import fire as fire
import pandas as pd
import requests
import tqdm

from databutler.utils import paths

_OwnerSlug = str
_KernelSlug = str


def _get_working_dir() -> str:
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "mining", "kaggle")


def _get_kernel_info(slug: str) -> Dict:
    text = requests.get(f"https://www.kaggle.com/{slug}").text
    d = next(i for i in re.compile(r"Kaggle.State.push\(({.*})\)").findall(text) if "runInfo" in i)
    d = json.loads(d)

    return d


def get_notebook_slugs() -> List[Tuple[_OwnerSlug, _KernelSlug]]:
    """
    Returns notebook slugs i.e. (owner-username, kernel-slug) tuples using the Meta-Kaggle dataset at
    https://www.kaggle.com/kaggle/meta-kaggle.

    Returns:
        (List[Tuple[_OwnerSlug, _KernelSlug]]): A list of tuples where the first element is the username of the
            owner and the second element is the kernel slug. The notebook's URL can then be constructed as
            https://www.kaggle.com/username/kernel-slug
    """
    #  Fetch all the useful data from the Meta-Kaggle dataset.
    kernels_csv_path = os.path.join(_get_working_dir(), "Kernels.csv")
    users_csv_path = os.path.join(_get_working_dir(), "Users.csv")
    kernel_versions_csv_path = os.path.join(_get_working_dir(), "KernelVersions.csv")

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


def download_metadata():
    slugs: List[Tuple[_OwnerSlug, _KernelSlug]] = list(get_notebook_slugs())

    with open(os.path.join(_get_working_dir(), "metadata.pkl"), "wb") as f:
        succ = fail = 0
        with tqdm.tqdm(slugs) as pbar:
            for owner_slug, kernel_slug in pbar:
                try:
                    metadata = _get_kernel_info("/".join(slugs[0]))

                    pickle.dump({
                        "owner_slug": owner_slug,
                        "kernel_slug": kernel_slug,
                        "metadata": metadata,
                    }, file=f)

                    f.flush()

                except:
                    fail += 1

                pbar.set_postfix(succ=succ, fail=fail)


if __name__ == "__main__":
    fire.Fire()
