import os
import random
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Optional, Tuple, Iterator

import attrs
import click
import fire
import tqdm

import numpy as np
import pandas as pd
from databutler.mining.static_pandas_mining.mining_utils import (
    find_library_usages,
    find_constants,
    replace_constants,
    DF_TYPE,
    SERIES_TYPE,
    GROUPBY_TYPES,
    extract_context,
    normalize_df_series_vars,
    normalize_call_args,
    normalize_col_accesses,
    templatize,
    get_mypy_cache_dir_path,
    MinedResult,
    get_created_mypy_cache_dir_paths,
    is_purely_df_or_series_like,
    find_single_use_expressions,
)
from databutler.pat import astlib
from databutler.pat.analysis.type_analysis.inference import run_mypy
from databutler.pat.analysis.type_analysis.mypy_types import SerializedMypyType
from databutler.utils import pickleutils, code as codeutils, multiprocess

JsonDict = Dict
MINING_RESULTS_FILE = "pandas_mining_results.pkl"
PROCESSED_KEYS_FILE = "pandas_mining_processed_keys.pkl"

KNOWN_ATTRS: Set[str] = {
    "loc",
    "iloc",
    "at",
    "iat",
    "str",
    "dt",
    "cat",
    "plot",
    "hist",
    "box",
    "kde",
    "area",
    "scatter",
    "hexbin",
}
MAX_STRING_CONSTANTS = 15


def get_num_string_constants(node: astlib.AstNode) -> int:
    return sum(1 for n in astlib.walk(node) if isinstance(n, astlib.SimpleString))


def prepare_mined_result(
    target: astlib.AstNode,
    code_ast: astlib.AstNode,
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
    lib_usages: Dict[astlib.Name, str],
    constants: Dict[astlib.BaseExpression, Any],
    kind: str,
    reference: str,
) -> Optional[MinedResult]:
    # import time
    # s = time.time()
    expr_type = inferred_types.get(target, None)
    true_exprs: List[astlib.BaseExpression] = list(
        astlib.iter_true_exprs(target, code_ast)
    )
    target_nodes = set(astlib.walk(target))
    free_vars: Set[astlib.Name] = {
        a.node
        for a in astlib.get_definitions_and_accesses(code_ast)[1]
        if all(d.enclosing_node not in target_nodes for d in a.definitions)
    }.intersection(true_exprs)

    def _fixup_metadata(node_mapping, _target):
        nonlocal inferred_types, lib_usages, true_exprs, free_vars
        inferred_types = {node_mapping.get(k, k): v for k, v in inferred_types.items()}
        lib_usages = {node_mapping.get(k, k): v for k, v in lib_usages.items()}
        _target_exprs = {
            n: idx
            for idx, n in enumerate(astlib.walk(_target))
            if isinstance(n, astlib.BaseExpression)
        }
        true_exprs = sorted(
            {node_mapping.get(n, n) for n in true_exprs}.intersection(
                _target_exprs.keys()
            ),
            key=lambda x: _target_exprs.get(x, 0),
        )
        free_vars = sorted(
            {
                node_mapping.get(n, n)
                for n in free_vars
                if isinstance(node_mapping.get(n, n), astlib.Name)
            }.intersection(true_exprs),
            key=lambda x: _target_exprs.get(x, 0),
        )

    #  Replace any constant expressions
    target, repl_map = replace_constants(target, true_exprs, free_vars, constants)
    _fixup_metadata(repl_map, target)
    # print("CONST REPL", astlib.to_code(target))

    #  Ignore if any undefined non-import, non-df, non-series, any-type variables
    context_vars = extract_context(target, free_vars, inferred_types, lib_usages)
    if context_vars is None:
        return None

    # if len(context_vars) != 0:
    #     print("CONTEXT_VARS", context_vars)
    #     print(astlib.to_code(target))

    target, repl_map = normalize_call_args(target, inferred_types)
    _fixup_metadata(repl_map, target)

    #  Normalize df and series variables (these will become parameters)
    target, df_vars, series_vars, repl_map = normalize_df_series_vars(
        target, true_exprs, free_vars, inferred_types
    )
    _fixup_metadata(repl_map, target)

    #  Convert attribute-based column accesses to subscript-based accesses.
    target, repl_map = normalize_col_accesses(target, true_exprs, inferred_types)
    for expr in astlib.walk(target):
        if isinstance(expr, astlib.SimpleString):
            true_exprs.append(expr)
    _fixup_metadata(repl_map, target)

    if get_num_string_constants(target) > MAX_STRING_CONSTANTS:
        return None

    #  Create templates for clustering
    template, template_vars_map = templatize(
        target, true_exprs, free_vars, inferred_types, lib_usages
    )
    # print("TEMPLATIZED", astlib.to_code(template), len(true_exprs))

    res = MinedResult(
        code=codeutils.normalize_code_fast(astlib.to_code(target)),
        template=codeutils.normalize_code_fast(astlib.to_code(template)),
        kind=kind,
        reference=reference,
        uid="",  # Will be set later
        expr_type=expr_type,
        type_map={
            codeutils.normalize_code_fast(astlib.to_code(k)): v
            for k, v in inferred_types.items()
        },
        df_vars=df_vars,
        series_vars=series_vars,
        template_vars=template_vars_map,
        extra_context_vars=context_vars,
        lib_usages={k.value: v for k, v in lib_usages.items()},
    )
    # print("-----", id(code_ast), time.time() - s)
    return res


def generic_mine_code(
    code: str,
    reference: str,
    base_uid: str,
    mypy_cache_path: Optional[str] = None,
) -> List[MinedResult]:
    result: List[MinedResult] = []
    code_ast = astlib.parse(code)
    _, inferred_types = run_mypy(code_ast, cache_dir=mypy_cache_path)

    lib_usages = find_library_usages(code_ast)
    constants = find_constants(code_ast)

    #  1. Find non-name expressions that result in a pandas dataframe, series, or groupby
    df_exprs: Set[astlib.BaseExpression] = set()
    series_exprs: Set[astlib.BaseExpression] = set()
    groupby_exprs: Set[astlib.BaseExpression] = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if node in inferred_types:
            if is_purely_df_or_series_like(inferred_types[node]):
                if inferred_types[node].equals(DF_TYPE):
                    df_exprs.add(node)
                    # print("DF", astlib.to_code(node))
                elif inferred_types[node].equals(SERIES_TYPE):
                    #  Check if it is an attribute and we have erroneously identified a column
                    if (
                        isinstance(node, astlib.Attribute)
                        and node.value in inferred_types
                        and inferred_types[node.value].equals(DF_TYPE)
                    ):
                        if hasattr(pd.DataFrame, node.attr.value):
                            continue
                        # print("DF", astlib.to_code(node))

                    series_exprs.add(node)
                    # print("SERIES", astlib.to_code(node))
            elif any(inferred_types[node].equals(i) for i in GROUPBY_TYPES):
                groupby_exprs.add(node)
                # print("GROUPBY", astlib.to_code(node))

    df_series_gpby_exprs = df_exprs | series_exprs | groupby_exprs

    #  2. Find pandas API calls (that may not yield a dataframe / series)
    api_usage_exprs: Set[astlib.BaseExpression] = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if (not isinstance(node, astlib.Call)) or node in df_series_gpby_exprs:
            continue

        #  Caller type if known, should be a callable
        caller_type = inferred_types.get(node.func, None)
        if (
            caller_type is not None
            and (not caller_type.is_any_type())
            and (not caller_type.is_callable_type())
        ):
            continue

        #  We are looking for a function call whose caller involves a dataframe/series/groupby
        if any(
            n in df_series_gpby_exprs
            for n in astlib.iter_true_exprs(node.func, code_ast)
        ):
            api_usage_exprs.add(node)

        #  Also watch out for things like pd.concat
        if any(
            isinstance(n, astlib.Name) and n in lib_usages and "pandas" in lib_usages[n]
            for n in astlib.iter_true_exprs(node.func, code_ast)
        ):
            api_usage_exprs.add(node)

    all_found_exprs = df_series_gpby_exprs | api_usage_exprs

    #  3. Find function calls / subscript accesses that take dataframe / series / groupby arguments that were not
    #  identified previously
    call_exprs_with_df_series_gpby_args = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if isinstance(node, astlib.Call) and node not in all_found_exprs:
            #  Caller type if known, should be a callable
            caller_type = inferred_types.get(node.func, None)
            if (
                caller_type is not None
                and (not caller_type.is_any_type())
                and (not caller_type.is_callable_type())
            ):
                continue

            #  Do not want expressions whose parents were found in the previous step
            if any(
                n in all_found_exprs for n in astlib.iter_parents(node.func, code_ast)
            ):
                continue

            if any(arg.value in df_series_gpby_exprs for arg in node.args):
                call_exprs_with_df_series_gpby_args.add(node)

    all_found_exprs.update(call_exprs_with_df_series_gpby_args)
    subscript_exprs_with_df_series_gpby_values = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if isinstance(node, astlib.Subscript) and node not in all_found_exprs:
            #  Do not want expressions whose parents were found in the previous step
            if any(
                n in all_found_exprs for n in astlib.iter_parents(node.value, code_ast)
            ):
                continue

            if any(child in all_found_exprs for child in astlib.iter_children(node)):
                subscript_exprs_with_df_series_gpby_values.add(node)

    #  Find attribute expressions that take dataframe / series / groupby values that were identified / ignored
    #  in the previous steps
    all_found_exprs.update(subscript_exprs_with_df_series_gpby_values)
    attr_access_exprs: Set[astlib.Attribute] = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if (
            isinstance(node, astlib.Attribute)
            and node not in all_found_exprs
            and node.value in df_series_gpby_exprs
        ):
            #  Ignore these by default
            if node.attr.value in KNOWN_ATTRS:
                continue

            #  Do not want expressions whose parents were found in the previous step
            if any(n in all_found_exprs for n in astlib.iter_parents(node, code_ast)):
                continue

            #  Parent cannot be another attribute or a function call
            parent = astlib.get_parent(node, code_ast)
            if isinstance(parent, astlib.Attribute) or isinstance(parent, astlib.Call):
                continue

            if node.value in df_series_gpby_exprs:
                attr_access_exprs.add(node)

    #  Eliminate accessor expressions.
    #  For example, do not count df['A'] > 10, when it is part of df[df['A'] > 10]
    for expr in df_exprs | series_exprs:
        if isinstance(expr, astlib.Subscript):
            for subscript_elem in expr.slice:
                if hasattr(subscript_elem.slice, "value"):
                    df_exprs.discard(subscript_elem.slice.value)
                    series_exprs.discard(subscript_elem.slice.value)

    for expr in df_exprs:
        if not isinstance(expr, astlib.Name):
            result.append(
                prepare_mined_result(
                    expr,
                    code_ast,
                    inferred_types,
                    lib_usages,
                    constants,
                    "DF_EXPR",
                    reference,
                )
            )

    for expr in series_exprs:
        if not isinstance(expr, astlib.Name):
            result.append(
                prepare_mined_result(
                    expr,
                    code_ast,
                    inferred_types,
                    lib_usages,
                    constants,
                    "SERIES_EXPR",
                    reference,
                )
            )

    for expr in api_usage_exprs:
        if not isinstance(expr, astlib.Name):
            result.append(
                prepare_mined_result(
                    expr,
                    code_ast,
                    inferred_types,
                    lib_usages,
                    constants,
                    "API_USAGE",
                    reference,
                )
            )

    for expr in call_exprs_with_df_series_gpby_args:
        result.append(
            prepare_mined_result(
                expr,
                code_ast,
                inferred_types,
                lib_usages,
                constants,
                "CALLED_W_PD_ARGS",
                reference,
            )
        )

    for expr in attr_access_exprs:
        result.append(
            prepare_mined_result(
                expr,
                code_ast,
                inferred_types,
                lib_usages,
                constants,
                "DF_SERIES_GROUPBY_ATTR_ACCESS",
                reference,
            )
        )

    for expr in subscript_exprs_with_df_series_gpby_values:
        result.append(
            prepare_mined_result(
                expr,
                code_ast,
                inferred_types,
                lib_usages,
                constants,
                "SUBSCRIPT_W_PD_ARGS",
                reference,
            )
        )

    result = [res for res in result if res is not None]

    #  Dedup by code
    result = list({res.code: res for res in result}.values())

    #  Remove multiple entries of DF1[STR1] (column-access) as they are usually too many in number
    col_acc_template = "DF1[STR1]"
    if any(res.template == col_acc_template for res in result):
        col_access_representative = next(
            res for res in result if res.template == col_acc_template
        )
        result = [
            res
            for res in result
            if (not res.template == col_acc_template)
            or res is col_access_representative
        ]

    #  Assign UIDs
    for idx, res in enumerate(result, 1):
        res.uid = f"{base_uid}:{idx}"

    return result


@attrs.define(eq=False, repr=False)
class MiningTask:
    normalized_code: str
    reference: str
    base_uid: str


def _run_mining_task_mp(
    args: Tuple[MiningTask, multiprocess.mp.Queue]
) -> List[MinedResult]:
    task, available_mypy_cache_paths = args
    try:
        cache_path = available_mypy_cache_paths.get(block=False)
        # print(f"\nReusing {cache_path}:{os.getpid()}\n", flush=True)
    except multiprocess.QueueEmptyException:
        cache_path = get_mypy_cache_dir_path(os.getpid())
        # print(f"\nCreated New {cache_path}:{os.getpid()}\n", flush=True)

    try:
        return generic_mine_code(
            task.normalized_code,
            task.reference,
            task.base_uid,
            mypy_cache_path=cache_path,
        )
    finally:
        #  This will likely not be executed if there is a timeout, so the cache path will be lost
        #  until the chunk is finished. That's a loss we will have to take.
        #  No gains in complicating the solution further.
        available_mypy_cache_paths.put(cache_path)


@attrs.define(eq=False, repr=False)
class BaseMiningCampaign(ABC):
    campaign_dir: str
    random_seed: int = attrs.field(default=42)

    def reset_random_seed(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @abstractmethod
    def nb_keys_iterator(self) -> Iterator[str]:
        pass

    @abstractmethod
    def get_tasks_for_keys(self, keys: List[str]) -> List[MiningTask]:
        pass

    @staticmethod
    def construct_mining_results_path(campaign_dir: str) -> str:
        return os.path.join(campaign_dir, MINING_RESULTS_FILE)

    @property
    def mining_results_path(self) -> str:
        return self.construct_mining_results_path(self.campaign_dir)

    @property
    def processed_keys_path(self) -> str:
        return os.path.join(self.campaign_dir, PROCESSED_KEYS_FILE)

    def get_already_processed_keys(self) -> Set[str]:
        if not os.path.exists(self.processed_keys_path):
            return set()
        with pickleutils.PickledMapReader(self.processed_keys_path) as reader:
            return set(reader.keys())

    def run(
        self,
        append: bool = False,
        num_processes: int = 2,
        chunk_size: int = 10000,
        timeout_per_notebook: int = 100,
        saving_frequency: int = 1000,
        num_notebooks: Optional[int] = None,
        start_idx: Optional[int] = None,
    ) -> None:
        campaign_dir: str = self.campaign_dir
        os.makedirs(campaign_dir, exist_ok=True)
        if os.path.exists(self.mining_results_path) and not append:
            if not click.confirm(
                "Overwrite existing mining results and restart from scratch?"
            ):
                print(
                    f"Cancelling... Use --append if you want to add to existing results."
                )
                return

            os.unlink(self.mining_results_path)
            if os.path.exists(self.processed_keys_path):
                os.unlink(self.processed_keys_path)

        self.reset_random_seed()
        all_nb_keys: List[str] = list(self.nb_keys_iterator())
        print(f"Found {len(all_nb_keys)} notebooks in total")
        already_processed_keys: Set[str] = self.get_already_processed_keys()
        print(f"Found {len(already_processed_keys)} already processed notebooks")
        keys_to_process: List[str] = [
            key for key in all_nb_keys if key not in already_processed_keys
        ]

        random.shuffle(keys_to_process)
        if num_notebooks is not None or start_idx is not None:
            num_notebooks = num_notebooks or len(keys_to_process)
            start_idx = start_idx or 0
            keys_to_process = keys_to_process[start_idx : start_idx + num_notebooks]
            print(f"Only considering {len(keys_to_process)} notebooks")
        else:
            print(f"Considering {len(keys_to_process)} notebooks")

        num_snippets_found = 0
        succ = exceptions = timeouts = other = 0

        available_mypy_cache_paths = multiprocess.generate_queue()
        og_cache_dirs: Set[str] = {
            get_mypy_cache_dir_path(i) for i in range(num_processes)
        }
        for cache_path in og_cache_dirs:
            available_mypy_cache_paths.put(cache_path)

        def _remove_mypy_cache_path(path: str):
            if os.path.exists(path):
                print(f"Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"WARNING: Did not find mypy cache path {path}")

        unique_code_so_far: Set[str] = set()

        with pickleutils.PickledMapWriter(
            self.mining_results_path, overwrite_existing=(not append)
        ) as writer, pickleutils.PickledMapWriter(
            self.processed_keys_path, overwrite_existing=(not append)
        ) as processed_keys_writer, open(
            os.path.join(campaign_dir, "mining_snippets_log.txt"), "w"
        ) as log_file:
            try:
                for idx in tqdm.tqdm(range(0, len(keys_to_process), chunk_size)):
                    chunk = keys_to_process[idx : idx + chunk_size]
                    tasks: List[Tuple[MiningTask, multiprocess.mp.Queue]] = [
                        (task, available_mypy_cache_paths)
                        for task in self.get_tasks_for_keys(chunk)
                    ]

                    try:
                        save_ctr = 0
                        mp_iter = multiprocess.run_tasks_in_parallel_iter(
                            _run_mining_task_mp,
                            tasks=tasks,
                            use_progress_bar=True,
                            num_workers=num_processes,
                            timeout_per_task=timeout_per_notebook,
                        )
                        for key, (task, _), result in zip(chunk, tasks, mp_iter):
                            if (
                                result.is_success()
                                and isinstance(result.result, list)
                                and len(result.result) > 0
                            ):
                                num_snippets_found += len(result.result)
                                for snippet in result.result:
                                    writer[snippet.uid] = snippet
                                    unique_code_so_far.add(snippet.code)

                            print(f"Processed {key}")
                            if result.is_success():
                                succ += 1
                                processed_keys_writer[key] = True
                            elif result.is_exception():
                                print(f"Failed for {task.reference}")
                                exceptions += 1
                                processed_keys_writer[key] = False
                            elif result.is_timeout():
                                print(f"Timed out for {task.reference}")
                                timeouts += 1
                                processed_keys_writer[key] = False
                            else:
                                other += 1
                                processed_keys_writer[key] = False

                            #  Make sure we save intermediate results. Saving frequency shouldn't be too high so as to
                            #  burden the file system.
                            save_ctr += 1
                            if save_ctr == saving_frequency:
                                save_ctr = 0
                                writer.flush()
                                processed_keys_writer.flush()

                        print(
                            f"\n-----\n"
                            f"Snippets found so far: {num_snippets_found}\n"
                            f"Success: {succ} Exceptions: {exceptions} Timeouts: {timeouts}"
                            f"\n-----\n"
                        )

                    finally:
                        writer.flush()
                        print("Cleaning up...")
                        #  Remove the non-og mypy cache paths
                        while not available_mypy_cache_paths.empty():
                            available_mypy_cache_paths.get()

                        created_cache_dirs = set(get_created_mypy_cache_dir_paths())
                        for path in created_cache_dirs - og_cache_dirs:
                            _remove_mypy_cache_path(path)

                        #  Requeue the og cache paths
                        for cache_path in og_cache_dirs:
                            available_mypy_cache_paths.put(cache_path)

                        for code in sorted(unique_code_so_far, key=len):
                            print(code.strip(), file=log_file)

                        log_file.flush()
                        unique_code_so_far.clear()

            finally:
                #  Remove the og mypy cache paths
                for path in get_created_mypy_cache_dir_paths():
                    _remove_mypy_cache_path(path)

        print("----------------------")
        print(f"Total Snippets Found: {num_snippets_found}")
        print("----------------------")

    def merge_mining_results(self, master_campaign_dir: str, *other_campaign_dirs: str):
        master_results_path = os.path.join(master_campaign_dir, MINING_RESULTS_FILE)
        other_results_paths: List[str] = [
            os.path.join(other_campaign_dir, MINING_RESULTS_FILE)
            for other_campaign_dir in other_campaign_dirs
        ]

        seen_templates: Set[str] = set()
        seen_uids: Set[str] = set()
        with pickleutils.PickledMapReader(master_results_path) as master_reader:
            for value in tqdm.tqdm(
                master_reader.values(),
                total=len(master_reader),
                desc="Reading master results",
            ):
                assert isinstance(value, MinedResult)
                seen_templates.add(value.template)
                seen_uids.add(value.uid)

        for path in other_results_paths:
            uids_to_add: Set[str] = set()
            with pickleutils.PickledMapReader(path) as reader:
                for value in tqdm.tqdm(
                    reader.values(),
                    total=len(reader),
                    desc=f"Reading results from {path}",
                ):
                    assert isinstance(value, MinedResult)
                    if value.template not in seen_templates:
                        uids_to_add.add(value.uid)

                print(f"Adding {len(uids_to_add)} uids from {path}")

                with pickleutils.PickledMapWriter(
                    master_results_path, overwrite_existing=False
                ) as writer:
                    for uid in tqdm.tqdm(
                        uids_to_add, total=len(uids_to_add), desc="Adding uids"
                    ):
                        item = reader[uid]
                        if uid in seen_uids:
                            item.uid = f"{uid}:duplicate_{os.path.basename(os.path.dirname(path))}"

                        writer[item.uid] = item
