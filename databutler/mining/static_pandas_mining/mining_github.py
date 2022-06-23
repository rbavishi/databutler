from typing import List, Iterator

import attrs
import fire

from databutler.mining.github.datalore_scraping import get_nb_reader
from databutler.mining.static_pandas_mining.mining_core import (
    BaseMiningCampaign,
    MiningTask,
)


@attrs.define(eq=False, repr=False)
class GithubMiningCampaign(BaseMiningCampaign):
    def nb_keys_iterator(self) -> Iterator[str]:
        with get_nb_reader(self.campaign_dir) as reader:
            yield from reader.keys()

    def get_tasks_for_keys(self, keys: List[str]) -> List[MiningTask]:
        tasks: List[MiningTask] = []
        with get_nb_reader(self.campaign_dir) as reader:
            for key in keys:
                nb_json = reader[key]
                reference = (
                    f"https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/{key}"
                )
                base_uid = key
                tasks.append(MiningTask(nb_json, reference, base_uid))

        return tasks


def campaign(campaign_dir: str) -> GithubMiningCampaign:
    return GithubMiningCampaign(campaign_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "campaign": campaign,
        }
    )
