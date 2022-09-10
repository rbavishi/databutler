import ast
import collections
import itertools
import os
import random
from typing import Iterator, Tuple, List, Set, Deque, Dict, Any

import astunparse
import attrs
import fire
import tqdm
import yaml
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util,
    models,
)
from torch.utils.data import DataLoader

from databutler.mining.static_pandas_mining.autodoc_preprocessing import (
    AutodocPreprocessing,
    PreprocessedItem,
)
from databutler.mining.static_pandas_mining.autodoc_result import (
    AutodocResult,
    NLDescription,
    AutodocDescription,
    CanonicalAutodocDescription,
)
from databutler.mining.static_pandas_mining.autodoc_strategies import (
    AutodocFewShotExample,
    DescriptionsGenerator,
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

    max_errors: int = 3
    too_many_errors_penalty: int = 50
    error_counts: Dict[str, int] = attrs.field(factory=dict)
    penalty_served_to_templates: Dict[int, Set[str]] = attrs.field(factory=dict)
    penalty_serving_template_to_iter_map: Dict[
        str, Iterator[PreprocessedItem]
    ] = attrs.field(factory=dict)

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

        self.penalty_served_to_templates = collections.defaultdict(set)
        self.error_counts = collections.defaultdict(int)

    def get_next_chunk(self) -> List[PreprocessedItem]:
        chunk: List[PreprocessedItem] = []
        max_length = len(self.active_item_iters)
        iter_worklist: Deque[
            Tuple[str, Iterator[PreprocessedItem]]
        ] = collections.deque()
        new_active_iters: List[Tuple[str, Iterator[PreprocessedItem]]] = []

        #  First check if any templates are up for re-evaluation after serving their penalty.
        for template in self.penalty_served_to_templates[0]:
            iter_worklist.append(
                (template, self.penalty_serving_template_to_iter_map[template])
            )

        iter_worklist.extend(self.active_item_iters)
        while len(iter_worklist) > max_length:
            self.item_iters_queue.appendleft(iter_worklist.pop())

        #  Decrease the penalty counter for the others
        self.penalty_served_to_templates = collections.defaultdict(
            set,
            {k - 1: v for k, v in self.penalty_served_to_templates.items() if k > 0},
        )

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

    def record_template_success(self, template: str):
        #  Reset the error counts
        self.error_counts[template] = 0
        #  Reset the penalty being served
        for templates in self.penalty_served_to_templates.values():
            if template in templates:
                templates.remove(template)
                self.penalty_served_to_templates[0].add(template)
                break

    def record_template_failure(self, template: str):
        #  Increment the error counts
        self.error_counts[template] += 1

        #  If errors reach the limit, remove from active iters and add to the penalty map
        if self.error_counts[template] >= self.max_errors:
            try:
                _, iter_ = self.active_item_iters[
                    next(
                        idx
                        for idx in range(len(self.active_item_iters))
                        if self.active_item_iters[idx][0] == template
                    )
                ]
                orig_len = len(self.active_item_iters)
                self.active_item_iters = [
                    elem for elem in self.active_item_iters if elem[0] != template
                ]
                while (
                    len(self.active_item_iters) < orig_len
                    and len(self.item_iters_queue) > 0
                ):
                    self.active_item_iters.append(self.item_iters_queue.popleft())

                self.penalty_served_to_templates[self.too_many_errors_penalty].add(
                    template
                )
                self.penalty_serving_template_to_iter_map[template] = iter_
                self.error_counts[template] = 0

            except StopIteration:
                pass

    def is_finished(self) -> bool:
        return len(self.active_item_iters) == 0


def get_param_types(param_code: str):
    func = ast.parse(param_code).body[0]
    assert isinstance(func, ast.FunctionDef)
    typ_ctr: Dict[str, int] = collections.defaultdict(int)
    for arg in func.args.args:
        if arg.annotation is None:
            typ_ctr["Any"] += 1
        else:
            typ_ctr[astunparse.unparse(arg.annotation).strip()] += 1

    return frozenset(typ_ctr.items())


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
        desc_gen = DescriptionsGenerator()
        autodoc_results: List[AutodocResult] = desc_gen.generate(
            snippets=snippets,
            few_shot_examples=self.few_shot,
        )

        for snippet, result in zip(snippets, autodoc_results):
            logger.opt(colors=True, raw=True).info(
                f"<g>{len(result.canonical_descs)}</g> canonical successful, "
                f"<r>{len(result.failed_descs)}</r> canonical failed, "
                f"<g>{len(result.additional_descs)}</g> additional successful "
                f"for code: <e>{snippet.code}</e>\n"
            )

        for snippet, result in zip(snippets, autodoc_results):
            logger_ = logger.opt(colors=True, raw=True)
            if not result.success:
                logger_.info(f"<r>Failed for {snippet.code}</r>\n")
                continue

            logger_.info(f"Code: <e>{snippet.code}</e>\n")
            logger_.info(f"Template: <e>{snippet.template}</e>\n")
            logger_.info("-------------\n")

            logger_.info(
                f"<m>{len(result.canonical_descs)} Canonical Descriptions:</m>\n"
            )
            for desc in result.canonical_descs:
                logger_.info(f"<m>Assistance Level: {desc.assistance_level}</m>\n")
                logger_.info("<m>{desc}</m>\n", desc=desc.desc.pretty_print())
                logger_.info(f"<m>Parameterized NL: {desc.parameterized_nl}</m>\n")
                logger_.info(f"<m>Parameterized Code:\n{desc.parameterized_code}</m>\n")
                logger_.info("-------------\n")

            logger_.info(
                f"<y>{len(result.additional_descs)} Additional Descriptions:</y>\n"
            )
            for desc in result.additional_descs:
                logger_.info(f"<y>Assistance Level: {desc.assistance_level}</y>\n")
                logger_.info("<y>{desc}</y>\n", desc=desc.desc.pretty_print())
                logger_.info("-------------\n")

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
                            additional_descs=autodoc_res.additional_descs,
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

                # chunk_snippets = [
                #     s
                #     for s in chunk_snippets
                #     if "drop" in s.code and "inplace" in s.code
                # ]

                for snippet in chunk_snippets:
                    logger.opt(raw=True).info(
                        f"Processing {snippet.uid}: {snippet.code} "
                        f"(support={preprocessing.template_support_dict[snippet.template]})\n"
                    )

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
                        chunk_builder.record_template_success(snippet.template)
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
                        chunk_builder.record_template_failure(snippet.template)

                    status.processed_uids.add(snippet.uid)

                pbar.update(len(status.processed_uids) - cur_processed)
                pbar.set_postfix(
                    succ=len(status.successful_uids), fail=len(status.failed_uids)
                )
                pbar.refresh()

                writer_success.flush()
                writer_failed.flush()

    def setup_elasticsearch_server(self):
        es_client = Elasticsearch([{"host": "localhost", "port": 9200}])

        index_name = "autodoc"
        #  Delete the index if it already exists
        if es_client.indices.exists(index=index_name):
            print(f"Deleting index {index_name}")
            es_client.indices.delete(index=index_name)

        with pickleutils.PickledMapReader(
            os.path.join(self.campaign_dir, AUTODOC_SUCCESS_PATH)
        ) as reader:
            #  Collect data to add in bulk
            data = []
            ctr = 0
            for uid, autodoc_result in tqdm.tqdm(reader.items(), total=len(reader)):
                assert isinstance(autodoc_result, AutodocResult)
                if autodoc_result.is_derived:
                    continue
                data.append({"index": {"_id": ctr, "_index": index_name}})
                doc_strs: List[str] = []
                doc_strs.extend(
                    {desc.parameterized_nl for desc in autodoc_result.canonical_descs}
                )

                for desc in autodoc_result.additional_descs:
                    doc_strs.append(desc.desc.primary_desc)

                data.append(
                    {
                        "uid": uid,
                        "doc": "* " + "\n* ".join(doc_strs),
                        "code": autodoc_result.code,
                        "canonical": autodoc_result.canonical_descs[
                            0
                        ].desc.primary_desc,
                    }
                )

                ctr += 1

        #  Add the data in bulk
        es_client.bulk(index=index_name, body=data)
        print(f"Added {ctr} documents to index")

    def start_elasticsearch_server(self):
        index_name = "autodoc"
        es_client = Elasticsearch([{"host": "localhost", "port": 9200}])
        #  Confirm the index exists
        if not es_client.indices.exists(index=index_name):
            raise Exception(f"Index {index_name} does not exist")

        #  Start the server
        while True:
            query = input("Query: ")
            if query == "exit":
                break

            res = es_client.search(
                index=index_name,
                body={
                    "query": {
                        "query_string": {
                            "query": query,
                            "fields": ["doc"],
                        }
                    }
                },
            )
            for hit in res["hits"]["hits"]:
                print(hit["_source"]["uid"])
                print(hit["_source"]["canonical"])
                print(hit["_source"]["code"])
                print("\n")

    def get_embedding_model(self):
        word_embedding_model = models.Transformer("microsoft/codebert-base")
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model

    def prepare_dataset_for_training_embeddings(
        self, run_id: str, per_equiv_class: int = 10
    ):
        autodoc_successes_path = os.path.join(self.campaign_dir, AUTODOC_SUCCESS_PATH)
        with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
            all_autodoc_results: List[AutodocResult] = list(
                res
                for res in tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
                if not res.is_derived
            )

        print(f"Found {len(all_autodoc_results)} non-derived autodoc results")

        #  We'll use the vanilla model to generate difficult cases where the data-points
        #  are ranked close by default.
        model = self.get_embedding_model()

        equiv_classes_canonical: List[List[Dict[str, Any]]] = []
        equiv_classes_additional: List[List[Dict[str, Any]]] = []
        for res in tqdm.tqdm(all_autodoc_results, desc="Preprocessing"):
            assert isinstance(res, AutodocResult)
            canonical = []
            additional = []

            all_param_types = {
                get_param_types(desc.parameterized_code) for desc in res.canonical_descs
            }

            for desc in res.canonical_descs:
                canonical.append(
                    {
                        "text": desc.desc.primary_desc,
                        "idx": len(equiv_classes_canonical),
                        "all_param_types": all_param_types,
                    }
                )
                canonical.append(
                    {
                        "text": desc.parameterized_nl,
                        "idx": len(equiv_classes_canonical),
                        "all_param_types": all_param_types,
                    }
                )

            for desc in res.additional_descs:
                additional.append(
                    {
                        "text": desc.desc.primary_desc,
                        "idx": len(equiv_classes_additional),
                        "all_param_types": all_param_types,
                    }
                )

            equiv_classes_canonical.append(canonical)
            equiv_classes_additional.append(additional)
            random.shuffle(equiv_classes_canonical[-1])
            random.shuffle(equiv_classes_additional[-1])

        canonical_flattened = sum(equiv_classes_canonical, [])
        additional_flattened = sum(equiv_classes_additional, [])

        #  Precompute embeddings
        embeddings_canonical = model.encode(
            [i["text"] for i in canonical_flattened],
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        embeddings_additional = model.encode(
            [i["text"] for i in additional_flattened],
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        correct_pairs: List[Tuple[str, str]] = []
        incorrect_pairs: List[Tuple[str, str]] = []
        for equiv_classes, embeddings, flattened in [
            (equiv_classes_canonical, embeddings_canonical, canonical_flattened),
            (equiv_classes_additional, embeddings_additional, additional_flattened),
        ]:
            all_indices: List[int] = list(range(len(equiv_classes)))
            idx_to_embeddings = collections.defaultdict(list)
            for embedding, item in zip(embeddings, flattened):
                idx_to_embeddings[item["idx"]].append(embedding)

            for idx, equiv_class in enumerate(
                tqdm.tqdm(equiv_classes, desc="Preparing data")
            ):
                if len(equiv_class) < 2:
                    continue

                candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                while len(candidates) < per_equiv_class:
                    candidates.extend(
                        itertools.islice(
                            itertools.combinations(equiv_class, 2),
                            per_equiv_class - len(candidates),
                        )
                    )

                random.shuffle(candidates)

                for s1, s2 in candidates:
                    correct_pairs.append((s1["text"], s2["text"]))

                #  Add mostly easy cases via random
                other_sample = random.sample(all_indices, per_equiv_class // 2)
                while idx in other_sample or any(
                    len(equiv_classes[i]) < 1 for i in other_sample
                ):
                    other_sample = random.sample(all_indices, per_equiv_class // 2)

                for other_idx in other_sample:
                    s1 = random.choice(equiv_class)
                    s2 = random.choice(equiv_classes[other_idx])
                    incorrect_pairs.append((s1["text"], s2["text"]))

                #  Add difficult cases via embedding similarity
                candidates.clear()
                increased = True
                while len(candidates) < (per_equiv_class // 2) and increased:
                    increased = False
                    orig_len = len(candidates)
                    #  First find the top-100 closest embeddings
                    item_idx, item = random.choice(list(enumerate(equiv_class)))
                    distances = util.cos_sim(
                        idx_to_embeddings[idx][item_idx], embeddings
                    )[0]
                    hard_cands = []
                    top_results = torch.topk(distances, k=100)
                    seen_uids: Set[str] = set()
                    for score, res_item_idx in zip(top_results[0], top_results[1]):
                        res_item = flattened[res_item_idx]
                        if res_item["idx"] == idx or res_item["idx"] in seen_uids:
                            continue

                        seen_uids.add(res_item["idx"])
                        if not item["all_param_types"].isdisjoint(
                            res_item["all_param_types"]
                        ):
                            continue

                        hard_cands.append((score, item, res_item))

                    hard_cands = sorted(hard_cands, key=lambda x: x[0], reverse=True)[
                        : per_equiv_class // 2
                    ]
                    for score, s1, s2 in hard_cands:
                        candidates.append((s1["text"], s2["text"]))
                        # print("Adding", s1["text"], s2["text"], score)
                        if len(candidates) >= (per_equiv_class // 2):
                            break

                    if len(candidates) > orig_len:
                        increased = True

                incorrect_pairs.extend(candidates)

        print("SIZES", len(correct_pairs), len(incorrect_pairs))
        pickleutils.smart_dump(
            [correct_pairs, incorrect_pairs],
            os.path.join(self.campaign_dir, f"embedding_train_data_{run_id}.pkl"),
        )

    def train_embeddings(
        self,
        run_id: str,
        model_name: str = "all-mpnet-base-v2",
        loss_type: str = "cosine",
        num_epochs: int = 5,
        batch_size: int = 8,
    ):
        correct_pairs, incorrect_pairs = pickleutils.smart_load(
            os.path.join(self.campaign_dir, f"embedding_train_data_{run_id}.pkl")
        )
        random.shuffle(correct_pairs)
        random.shuffle(incorrect_pairs)

        # model = SentenceTransformer(model_name)
        model_name = "codebert-base"
        word_embedding_model = models.Transformer("microsoft/codebert-base")
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_examples = [
            *(
                InputExample(texts=[s1, s2], label=1)
                for s1, s2 in correct_pairs[: int(len(correct_pairs) * 0.8)]
            ),
            *(
                InputExample(texts=[s1, s2], label=0)
                for s1, s2 in incorrect_pairs[: int(len(incorrect_pairs) * 0.8)]
            ),
        ]

        test_examples = [
            *(
                InputExample(texts=[s1, s2], label=1)
                for s1, s2 in correct_pairs[int(len(correct_pairs) * 0.8) :]
            ),
            *(
                InputExample(texts=[s1, s2], label=0)
                for s1, s2 in incorrect_pairs[int(len(incorrect_pairs) * 0.8) :]
            ),
        ]

        if loss_type == "contrastive":
            train_loss = losses.ContrastiveLoss(model=model)
            output_path = os.path.join(
                self.campaign_dir, f"embedding_model_contrastive_{run_id}.{model_name}"
            )
        elif loss_type == "cosine":
            train_loss = losses.CosineSimilarityLoss(model=model)
            output_path = os.path.join(
                self.campaign_dir, f"embedding_model_cosine_{run_id}.{model_name}"
            )
            for ex in itertools.chain(train_examples, test_examples):
                ex.label = float(ex.label)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            test_examples, show_progress_bar=True
        )

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size
        )

        model.fit(
            [(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            show_progress_bar=True,
            save_best_model=True,
            output_path=output_path,
        )

    def prepare_search_engine(
        self,
        run_id: str,
        model_name: str = "codebert-base",
        loss_type: str = "cosine",
    ):
        model_path = os.path.join(
            self.campaign_dir, f"embedding_model_{loss_type}_{run_id}.{model_name}"
        )
        model = SentenceTransformer(model_path)

        autodoc_successes_path = os.path.join(self.campaign_dir, AUTODOC_SUCCESS_PATH)
        with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
            all_autodoc_results: List[AutodocResult] = list(
                res
                for res in tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
                if not res.is_derived
            )

        corpus: List[Dict] = []
        for res in all_autodoc_results:
            assert isinstance(res, AutodocResult)
            param_nl = ""
            param_code = ""
            for desc in res.canonical_descs:
                corpus.append(
                    {
                        "text": desc.desc.primary_desc,
                        "nl": desc.desc.primary_desc,
                        "code": desc.target_code,
                        "uid": res.uid,
                        "param_nl": desc.parameterized_nl,
                        "param_code": desc.parameterized_code,
                    }
                )
                param_nl = desc.parameterized_nl
                param_code = desc.parameterized_code

            for desc in res.additional_descs:
                corpus.append(
                    {
                        "text": desc.desc.primary_desc,
                        "nl": res.canonical_descs[0].desc.primary_desc,
                        "code": desc.target_code,
                        "uid": res.uid,
                        "param_nl": param_nl,
                        "param_code": param_code,
                    }
                )

        print(f"Found {len(corpus)} items")

        embeddings = model.encode(
            [item["text"] for item in corpus],
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        print("Finished generating embeddings")

        embeddings_path = os.path.join(
            model_path,
            f"search_engine_embeddings_{loss_type}_{run_id}.{model_name}.pkl",
        )
        pickleutils.smart_dump((corpus, embeddings), embeddings_path)
        print(f"Saved corpus and embeddings to {embeddings_path}")

    def start_search_engine(
        self,
        run_id: str,
        model_name: str = "codebert-base",
        loss_type: str = "cosine",
    ):
        model_path = os.path.join(
            self.campaign_dir, f"embedding_model_{loss_type}_{run_id}.{model_name}"
        )
        model = SentenceTransformer(model_path)
        # model = SentenceTransformer(model_name)
        embeddings_path = os.path.join(
            model_path,
            f"search_engine_embeddings_{loss_type}_{run_id}.{model_name}.pkl",
        )
        corpus, embeddings = pickleutils.smart_load(embeddings_path)
        print(f"Loaded corpus and embeddings from {model_path}")

        while True:
            query = input("Query: ")
            query_embedding = model.encode(
                query, show_progress_bar=False, convert_to_tensor=True
            )
            distances = util.cos_sim(query_embedding, embeddings)[0]

            top_results = torch.topk(distances, k=100)
            seen_code: Set[str] = set()
            ctr = 0
            for score, idx in zip(top_results[0], top_results[1]):
                if corpus[idx]["uid"] in seen_code:
                    continue

                seen_code.add(corpus[idx]["uid"])
                print(corpus[idx]["param_nl"], "(Score: {:.4f})".format(score))
                print(corpus[idx]["param_code"])
                print("---")
                ctr += 1

                if ctr == 10:
                    break

            print("\n----------\n")

    def debug_search_engine(
        self,
        run_id: str,
        model_name: str = "all-mpnet-base-v2",
        loss_type: str = "contrastive",
    ):
        model_path = os.path.join(
            self.campaign_dir, f"embedding_model_{loss_type}_{run_id}.{model_name}"
        )
        model = SentenceTransformer(model_path)
        corpus, embeddings = pickleutils.smart_load(
            os.path.join(model_path, f"search_engine_embeddings_{run_id}.pkl")
        )
        print(f"Loaded corpus and embeddings from {model_path}")

        while True:
            query1 = input("Query1: ")
            query1_embedding = model.encode(
                query1, show_progress_bar=False, convert_to_tensor=True
            )

            query2 = input("Query2: ")
            query2_embedding = model.encode(
                query2, show_progress_bar=False, convert_to_tensor=True
            )

            #  Print the distance
            print(util.cos_sim(query1_embedding, query2_embedding)[0])
            print("\n----------\n")

    def info(self) -> None:
        """Display detailed information about snippets processed, success rates etc."""
        preprocessing = self.run_preprocessing()
        status = AutodocStatus.from_campaign_dir(self.campaign_dir, preprocessing)
        print(f"Found {len(status.all_uids)} UIDs to process")
        print(
            f"Already processed {len(status.processed_uids)} UIDs "
            f"({len(status.successful_uids)} successful, {len(status.failed_uids)} failed)"
        )

        with pickleutils.PickledMapReader(
            self.mining_results_path
        ) as mining_results_reader, pickleutils.PickledMapReader(
            os.path.join(self.campaign_dir, AUTODOC_SUCCESS_PATH)
        ) as success_reader, pickleutils.PickledMapReader(
            os.path.join(self.campaign_dir, AUTODOC_FAILURE_PATH)
        ) as failed_reader:
            num_derived = num_nonderived = 0
            has_additional_descriptions = 0
            successful_templates = set()
            failed_templates = set()
            did_not_have_additional_descriptions = set()
            for uid in tqdm.tqdm(status.successful_uids, desc="Processing successful"):
                result: AutodocResult = success_reader[uid]
                successful_templates.add(result.template)
                if result.is_derived:
                    num_derived += 1
                else:
                    num_nonderived += 1

                if result.is_derived:
                    #  No further use of these
                    continue

                if len(result.additional_descs) > 0:
                    has_additional_descriptions += 1
                else:
                    did_not_have_additional_descriptions.add(result.code)

            for uid in tqdm.tqdm(status.failed_uids, desc="Processing failed"):
                result: AutodocResult = failed_reader[uid]
                failed_templates.add(result.template)

        print(f"{num_derived} derived, {num_nonderived} nonderived")
        print(
            f"{has_additional_descriptions} out of {num_nonderived} have additional descriptions"
        )
        print(f"{len(successful_templates)} successful templates")
        print(f"{len(failed_templates)} failed templates")

        with open(os.path.join(self.campaign_dir, "no_additional_descs.txt"), "w") as f:
            for code in did_not_have_additional_descriptions:
                f.write(code + "\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    fire.Fire(AutodocCampaign)
