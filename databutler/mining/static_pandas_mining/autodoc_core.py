import collections
import os
from typing import Iterator, Tuple, List, Set, Deque, Dict

import attrs
import fire
import tqdm
import yaml

from databutler.mining.static_pandas_mining.autodoc_preprocessing import (
    AutodocPreprocessing,
    PreprocessedItem,
)
from databutler.mining.static_pandas_mining.autodoc_result import AutodocResult
from databutler.mining.static_pandas_mining.autodoc_strategies import (
    AutodocFewShotExample,
    CanonicalDescriptionsGenerator,
    NLDescription,
    AutodocDescription,
    CanonicalAutodocDescription,
    normalize_code_results,
)
from databutler.mining.static_pandas_mining.autodoc_utils import find_instantiation_map
from databutler.mining.static_pandas_mining.mining_core import BaseMiningCampaign
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import pickleutils, code as codeutils
from databutler.utils.caching import cached_property
from databutler.utils.logging import logger

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
    def from_campaign_dir(
        campaign_dir: str, preprocessing: AutodocPreprocessing
    ) -> "AutodocStatus":
        success_path = os.path.join(campaign_dir, AUTODOC_SUCCESS_PATH)
        failure_path = os.path.join(campaign_dir, AUTODOC_FAILURE_PATH)

        all_uids: Set[str] = {item.key for item in preprocessing.items}
        successful_uids: Set[str] = set()
        failed_uids: Set[str] = set()

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


@attrs.define(eq=False, repr=False, slots=False)
class AutodocCampaign:
    campaign_dir: str
    path_few_shot: str

    @cached_property
    def few_shot(self) -> List[AutodocFewShotExample]:
        with open(self.path_few_shot, "r") as f:
            return AutodocFewShotExample.deserialize(yaml.full_load(f))

    @property
    def mining_results_path(self) -> str:
        return BaseMiningCampaign.construct_mining_results_path(self.campaign_dir)

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

    def process_snippets(self, snippets: List[MinedResult]) -> List[AutodocResult]:
        #  First try to generate canonical descriptions
        canonical_gen = CanonicalDescriptionsGenerator()
        descs_list: List[List[AutodocDescription]] = canonical_gen.generate(
            snippets=snippets,
            few_shot_examples=self.few_shot,
        )

        autodoc_results: List[AutodocResult] = []
        for snippet, descs in zip(snippets, descs_list):
            success_descs = []
            failed_descs = []
            for desc in descs:
                (
                    success_descs
                    if (desc.equivalent and desc.parameterized)
                    else failed_descs
                ).append(desc)

            autodoc_results.append(
                AutodocResult(
                    uid=snippet.uid,
                    code=snippet.code,
                    template=snippet.template,
                    success=len(success_descs) > 0,
                    canonical_descs=success_descs,
                    #  We will populate this later
                    additional_descs=[],
                    failed_descs=failed_descs,
                )
            )

            logger.opt(colors=True, raw=True).info(
                f"<g>{len(success_descs)}</g> successful and <r>{len(failed_descs)}</r> failed "
                f"for code: <e>{snippet.code}</e>\n"
            )

        return autodoc_results

    def attempt_autodoc_transfer(
        self,
        autodoc_res: AutodocResult,
        items: List[PreprocessedItem],
        reader: pickleutils.PickledMapReader,
    ) -> List[AutodocResult]:
        #  Given a parameterization, try to apply it to other snippets that share the same template.
        #  Specifically, check if the replacing the parameter usages with some expressions yields the target code.
        transferred_results: List[AutodocResult] = []

        parameterizations: List[Dict] = []
        for desc in autodoc_res.canonical_descs:
            param_code = desc.parameterized_code
            param_nl = desc.parameterized_nl
            param_code_ast = astlib.parse(param_code)
            param_func = next(astlib.iter_body_stmts(param_code_ast))
            expr = next(astlib.iter_body_stmts(param_func)).value

            expr_code = codeutils.normalize_code_fast(astlib.to_code(expr))
            expr_code = normalize_code_results(
                [expr_code], set(), replace_singleton_lists=False
            )[0]
            expr = astlib.parse_expr(expr_code)

            func_params: List[str] = [
                param.name.value for param in param_func.params.params
            ]
            parameterizations.append(
                {
                    "param_code": param_code,
                    "param_nl": param_nl,
                    "parameters": func_params,
                    "expr": expr,
                    "assistance_level": desc.assistance_level,
                    "nl_desc": desc.desc,
                }
            )

        succ = fail = 0
        with tqdm.tqdm(items) as pbar:
            for item in pbar:
                snippet = reader[item.key]
                snippet_code = normalize_code_results(
                    [snippet.code], set(snippet.df_vars), replace_singleton_lists=False
                )[0]

                found: List[CanonicalAutodocDescription] = []
                for param in parameterizations:
                    expr = param["expr"]
                    func_params = param["parameters"]
                    param_nl = param["param_nl"]
                    repl_map = find_instantiation_map(
                        expr, astlib.parse_expr(snippet_code)
                    )
                    if all(
                        isinstance(node, astlib.Name) for node in repl_map.keys()
                    ) and {node.value for node in repl_map.keys()}.issubset(
                        func_params
                    ):
                        #  Prepare the autodoc result for this snippet
                        nl_instantiation_map = {
                            f"[{param}]": f'"{param}"' for param in func_params
                        }
                        nl_instantiation_map.update(
                            {
                                f"[{k.value}]": astlib.to_code(v)
                                for k, v in repl_map.items()
                            }
                        )
                        instantiated_nl = param_nl
                        for k, v in nl_instantiation_map.items():
                            instantiated_nl = instantiated_nl.replace(k, v)

                        new_desc = NLDescription(
                            primary_desc=instantiated_nl,
                            auxiliary_descs=param["nl_desc"].auxiliary_descs,
                            context=param["nl_desc"].context,
                        )

                        found.append(
                            CanonicalAutodocDescription(
                                equivalent=True,
                                parameterized=True,
                                target_code=snippet_code,
                                target_template=snippet.template,
                                generated_code=snippet_code,
                                assistance_level=param["assistance_level"],
                                desc=new_desc,
                                parameterized_code=param["param_code"],
                                parameterized_nl=param["param_nl"],
                            )
                        )

                if len(found) > 0:
                    transferred_results.append(
                        AutodocResult(
                            uid=snippet.uid,
                            code=snippet.code,
                            template=snippet.template,
                            success=True,
                            canonical_descs=found,
                            additional_descs=[],
                            failed_descs=[],
                            is_derived=True,
                        )
                    )
                    succ += 1
                else:
                    fail += 1

                pbar.set_postfix(succ=succ, fail=fail)

        # for res in transferred_results:
        #     print(res.code)
        #     for desc in res.canonical_descs:
        #         print(desc.desc.primary_desc)

        return transferred_results

    def run(self, batch_size: int) -> None:
        preprocessing = self.run_preprocessing()
        print(
            f"Preprocessing done, found {len(preprocessing.template_processing_order)} templates"
        )

        status = AutodocStatus.from_campaign_dir(self.campaign_dir, preprocessing)
        print(f"Found {len(status.all_uids)} UIDs to process")
        print(
            f"Already processed {len(status.processed_uids)} UIDs "
            f"({len(status.successful_uids)} successful, {len(status.failed_uids)} failed)"
        )

        chunk_builder = AutodocChunkBuilder(preprocessing=preprocessing, status=status)
        chunk_builder.init(batch_size=batch_size)
        logger.add(os.path.join(self.campaign_dir, "autodoc.log"), level="DEBUG")

        with pickleutils.PickledMapReader(
            self.mining_results_path
        ) as mining_results_reader, pickleutils.PickledMapWriter(
            os.path.join(self.campaign_dir, AUTODOC_SUCCESS_PATH),
            overwrite_existing=False,
        ) as writer_success, pickleutils.PickledMapWriter(
            os.path.join(self.campaign_dir, AUTODOC_FAILURE_PATH),
            overwrite_existing=False,
        ) as writer_failed, tqdm.tqdm(
            total=len(status.all_uids)
        ) as pbar:
            pbar.update(len(status.processed_uids))
            pbar.refresh()

            while not chunk_builder.is_finished():
                cur_processed = len(status.processed_uids)
                chunk = chunk_builder.get_next_chunk()
                if len(chunk) == 0:
                    break

                logger.info(f"Processing chunk of size {len(chunk)}")
                chunk_snippets: List[MinedResult] = [
                    mining_results_reader[entry.key] for entry in chunk
                ]

                for snippet in chunk_snippets:
                    logger.opt(raw=True).info(
                        f"Processsing {snippet.uid}: {snippet.code}\n"
                    )

                # chunk_snippets = [s for s in chunk_snippets if "Glucose" in s.code and "replace" in s.code]
                # for s in chunk_snippets:
                #     print(s.extra_context_vars, s.template)
                try:
                    autodoc_results = self.process_snippets(chunk_snippets)
                except Exception as e:
                    logger.exception(e)
                    continue

                for snippet, autodoc_res in zip(chunk_snippets, autodoc_results):
                    if autodoc_res.success:
                        writer_success[snippet.uid] = autodoc_res
                        status.successful_uids.add(snippet.uid)
                        key = autodoc_res.uid
                        other_items = [
                            item
                            for item in preprocessing.template_to_items[
                                snippet.template
                            ]
                            if item.key not in status.processed_uids and item.key != key
                        ]

                        if len(other_items) > 0:
                            logger.info(f"Attempting transfer for: {snippet.code}")
                            transferred_results = self.attempt_autodoc_transfer(
                                autodoc_res, other_items, mining_results_reader
                            )
                            for transferred_result in transferred_results:
                                writer_success[
                                    transferred_result.uid
                                ] = transferred_result
                                status.successful_uids.add(transferred_result.uid)
                                status.processed_uids.add(transferred_result.uid)

                            logger.info(
                                f"Transferred {len(transferred_results)} results"
                            )

                    else:
                        writer_failed[snippet.uid] = autodoc_res
                        status.failed_uids.add(snippet.uid)

                    status.processed_uids.add(snippet.uid)

                pbar.update(len(status.processed_uids) - cur_processed)
                pbar.set_postfix(
                    succ=len(status.successful_uids), fail=len(status.failed_uids)
                )
                pbar.refresh()

                writer_success.flush()
                writer_failed.flush()


if __name__ == "__main__":
    fire.Fire(AutodocCampaign)
