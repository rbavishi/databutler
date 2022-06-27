import collections
from typing import Dict, List, Iterator, Tuple

import attrs
import tqdm

from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.utils.caching import cached_property


@attrs.define(eq=False, repr=False, slots=False)
class PreprocessedItem:
    key: str
    code: str
    template: str
    support: int

    @staticmethod
    def from_json(json_dict: dict) -> "PreprocessedItem":
        return PreprocessedItem(
            key=json_dict["key"],
            code=json_dict["code"],
            template=json_dict["template"],
            support=json_dict["support"],
        )

    def to_json(self) -> Dict:
        return {
            "key": self.key,
            "code": self.code,
            "template": self.template,
            "support": self.support,
        }


@attrs.define(eq=False, repr=False, slots=False)
class AutodocPreprocessing:
    items: List[PreprocessedItem]

    @staticmethod
    def from_results(
        results_iter: Iterator[Tuple[str, MinedResult]]
    ) -> "AutodocPreprocessing":
        preprocessed_items: List[PreprocessedItem] = []
        code_counter: Dict[str, int] = collections.Counter()
        for key, item in tqdm.tqdm(results_iter, desc="Preprocessing"):
            code_counter[item.code] += 1
            preprocessed_items.append(
                PreprocessedItem(
                    key=key,
                    code=item.code,
                    template=item.template,
                    support=0,
                )
            )

        #  Fix the support of each item.
        for item in preprocessed_items:
            item.support = code_counter[item.code]

        return AutodocPreprocessing(
            items=preprocessed_items,
        )

    @staticmethod
    def from_json(json_dict: dict) -> "AutodocPreprocessing":
        return AutodocPreprocessing(
            items=[PreprocessedItem.from_json(item) for item in json_dict["items"]],
        )

    def to_json(self) -> Dict:
        return {
            "items": [item.to_json() for item in self.items],
        }

    @cached_property
    def template_support_dict(self) -> Dict[str, int]:
        result = collections.defaultdict(int)
        for item in self.items:
            result[item.template] += item.support

        return result

    @cached_property
    def template_processing_order(self) -> List[str]:
        #  Sort the templates by support, in descending order.
        return sorted(
            self.template_support_dict.keys(),
            key=self.template_support_dict.get,
            reverse=True,
        )

    @cached_property
    def template_to_items(self) -> Dict[str, List[PreprocessedItem]]:
        result = collections.defaultdict(list)
        for item in self.items:
            result[item.template].append(item)

        return result
