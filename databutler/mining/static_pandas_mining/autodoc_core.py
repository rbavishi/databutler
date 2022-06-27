import json
import os
import collections
from typing import Iterator, Tuple, List, Set, Deque

import attrs
import fire

from databutler.mining.static_pandas_mining.autodoc_preprocessing import (
    AutodocPreprocessing,
    PreprocessedItem,
)
from databutler.mining.static_pandas_mining.mining_core import BaseMiningCampaign
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.utils import pickleutils
from databutler.utils.caching import cached_property
from databutler.mining.static_pandas_mining.autodoc_strategies import (
    AutodocFewShotExample,
)


AUTODOC_SUCCESS_PATH = "autodoc_success.pkl"
AUTODOC_FAILURE_PATH = "autodoc_failure.pkl"


@attrs.define(eq=False, repr=False)
class AutodocStatus:
    """Tracks the current status of the autodoc campaign - number of UIDs processed, successful, failures etc."""

    all_uids: Set[str]
    processed_uids: Set[str]
    successful_uids: Set[str]
    failed_uids: Set[str]

    @staticmethod
    def from_campaign_dir(campaign_dir: str) -> "AutodocStatus":
        mining_results_path = BaseMiningCampaign.construct_mining_results_path(
            campaign_dir
        )
        success_path = os.path.join(campaign_dir, AUTODOC_SUCCESS_PATH)
        failure_path = os.path.join(campaign_dir, AUTODOC_FAILURE_PATH)

        all_uids: Set[str] = set()
        successful_uids: Set[str] = set()
        failed_uids: Set[str] = set()

        if os.path.exists(mining_results_path):
            with pickleutils.PickledMapReader(mining_results_path) as reader:
                all_uids.update(reader.keys())

        if os.path.exists(success_path):
            with pickleutils.PickledMapReader(success_path) as reader:
                successful_uids.update(reader.keys())

        if os.path.exists(failure_path):
            with pickleutils.PickledMapReader(failure_path) as reader:
                failed_uids.update(reader.keys())

        return AutodocStatus(
            all_uids=all_uids,
            processed_uids=successful_uids | failed_uids,
            successful_uids=successful_uids,
            failed_uids=failed_uids,
        )


@attrs.define(eq=False, repr=False)
class AutodocChunkBuilder:
    """
    Builder for chunks to run autodoc on together.
    """

    preprocessing: AutodocPreprocessing
    status: AutodocStatus

    item_iters_queue: Deque[Tuple[str, Iterator[PreprocessedItem]]] = attrs.field(
        init=False
    )
    active_item_iters: List[Tuple[str, Iterator[PreprocessedItem]]] = attrs.field(
        init=False
    )

    def init(self, batch_size: int) -> None:
        self.item_iters_queue = collections.deque()
        for template in self.preprocessing.template_processing_order:
            self.item_iters_queue.append(
                (template, iter(self.preprocessing.template_to_items[template]))
            )

        self.active_item_iters = []
        while (len(self.active_item_iters) < batch_size) and (
            len(self.item_iters_queue) > 0
        ):
            self.active_item_iters.append(self.item_iters_queue.popleft())

    def get_next_chunk(self) -> List[PreprocessedItem]:
        chunk: List[PreprocessedItem] = []
        iter_worklist: Deque[
            Tuple[str, Iterator[PreprocessedItem]]
        ] = collections.deque(self.active_item_iters)
        new_active_iters: List[Tuple[str, Iterator[PreprocessedItem]]] = []

        while len(iter_worklist) > 0:
            template, iter_ = iter_worklist.popleft()
            for entry in iter_:
                if entry.key not in self.status.processed_uids:
                    chunk.append(entry)
                    new_active_iters.append((template, iter_))
                    break
            else:
                if len(self.item_iters_queue) > 0:
                    iter_worklist.append(self.item_iters_queue.popleft())

        self.active_item_iters.clear()
        self.active_item_iters.extend(new_active_iters)

        return chunk

    def is_finished(self) -> bool:
        return len(self.active_item_iters) == 0


@attrs.define(eq=False, repr=False)
class AutodocCampaign:
    campaign_dir: str
    path_few_shot: str

    @cached_property
    def few_shot(self) -> AutodocFewShotExample:
        with open(self.path_few_shot, "r") as f:
            return AutodocFewShotExample.from_json(json.load(f))

    def mining_results_iter(self) -> Iterator[Tuple[str, MinedResult]]:
        path = BaseMiningCampaign.construct_mining_results_path(self.campaign_dir)
        with pickleutils.PickledMapReader(path) as reader:
            for key, value in reader.items():
                yield key, value

    def get_preprocessing_path(self) -> str:
        #  Pickle is usually faster than JSON
        return os.path.join(self.campaign_dir, "autodoc_preprocessing.pkl")

    def run_preprocessing(self) -> AutodocPreprocessing:
        path = self.get_preprocessing_path()
        if os.path.exists(path):
            return pickleutils.smart_load(path)

        preprocessing = AutodocPreprocessing.from_results(self.mining_results_iter())
        #  Pickle is usually faster than JSON
        pickleutils.smart_dump(preprocessing, path)
        return preprocessing

    def run(self, batch_size: int) -> None:
        status = AutodocStatus.from_campaign_dir(self.campaign_dir)

        print(f"Found {len(status.all_uids)} UIDs to process")
        print(
            f"Already processed {len(status.processed_uids)} UIDs "
            f"({len(status.successful_uids)} successful, {len(status.failed_uids)} failed)"
        )

        preprocessing = self.run_preprocessing()
        print(
            f"Preprocessing done, found {len(preprocessing.template_processing_order)} templates"
        )
        chunk_builder = AutodocChunkBuilder(preprocessing=preprocessing, status=status)
        chunk_builder.init(batch_size=batch_size)

        while not chunk_builder.is_finished():
            chunk = chunk_builder.get_next_chunk()
            print(f"Processing chunk of size {len(chunk)}")
            for item in chunk:
                print(f"Processing {item.key} (support={item.support})")
                print(item.template)


if __name__ == "__main__":
    fire.Fire(AutodocCampaign)
