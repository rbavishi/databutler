from typing import List, Iterator, Optional

import attrs
import fire

from databutler.mining.kaggle_tools.notebooks import utils as nb_utils
from databutler.mining.kaggle_tools.notebooks.notebook import KaggleNotebook
from databutler.mining.static_pandas_mining.mining_core import (
    BaseMiningCampaign,
    MiningTask,
    MinedResult,
    generic_mine_code,
)
from databutler.pat import astlib
from databutler.utils import code as codeutils


def mine_notebook(owner: str, slug: str) -> None:
    nb = KaggleNotebook.from_raw_data(
        owner, slug, nb_utils.retrieve_notebook_data(owner, slug)
    )
    normalized_code = codeutils.normalize_code_fast(astlib.to_code(nb.get_astlib_ast()))
    reference = f"https://kaggle.com/{owner}/{slug}"
    base_uid = f"{owner}/{slug}"

    results: List[MinedResult] = generic_mine_code(normalized_code, reference, base_uid)
    for result in results:
        print(result.prettify())

    for result in sorted(results, key=lambda r: len(r.code), reverse=True):
        print(result.code)

    print(f"Mined {len(results)} snippets.")


@attrs.define(eq=False, repr=False)
class KaggleMiningCampaign(BaseMiningCampaign):
    def nb_keys_iterator(self) -> Iterator[str]:
        with nb_utils.get_local_nb_data_storage_reader() as reader:
            for key in reader.keys():
                yield key

    def get_tasks_for_keys(self, keys: List[str]) -> List[MiningTask]:
        tasks: List[MiningTask] = []
        with nb_utils.get_local_nb_data_storage_reader() as reader:
            for key in keys:
                owner, slug = key
                nb = KaggleNotebook.from_raw_data(owner, slug, raw_data=reader[key])
                nb_json = nb.source_code
                reference = f"https://kaggle.com/{owner}/{slug}"
                base_uid = f"{owner}/{slug}"
                tasks.append(MiningTask(nb_json, reference, base_uid))

        return tasks


def campaign(campaign_dir: str) -> KaggleMiningCampaign:
    return KaggleMiningCampaign(campaign_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "mine_notebook": mine_notebook,
            "campaign": campaign,
        }
    )
