import ast
import collections
import itertools
import os
import random
import shutil
from typing import Optional, List, Dict, Tuple, Set, Deque, Iterator

import fire
import tqdm
import yaml
import numpy as np
from simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration

import pandas as pd
from databutler.datana.generic.autodoc import code2nl, nl2code
from databutler.datana.generic.autodoc.few_shot import FewShotExampleCodeAndNL
from databutler.datana.generic.autoparameterization.few_shot import (
    FewShotExampleParameterization,
)
from databutler.datana.generic.autoparameterization.parameterizers import (
    SimpleParameterizer,
    ParameterizationTask,
)
from databutler.mining.static_pandas_mining.autodoc_utils import (
    AutodocFewShotExample,
    AutodocDescription,
    AutodocResult,
    normalize_code_for_comparison,
    get_few_shot_example_path,
    generate_nl_based_parameterization,
    attempt_parameterization_application,
    Parameterization,
    generate_llm_based_parameterization,
)
from databutler.mining.static_pandas_mining.mining import MINING_RESULTS_FILE
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import pickleutils, langmodels

from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    losses,
    InputExample,
    evaluation,
    util,
)
from torch.utils.data import DataLoader
import torch

ENGINE = "code-davinci-002"
PREPROCESSING_RESULTS_FILE = "pandas_mining_preprocessed.pkl"
AUTODOC_SUCCESSES_PATH = "autodoc_results.pkl"
AUTODOC_FAILURES_FILE = "autodoc_failures.pkl"
MAX_TEMPLATE_FAILURES = 5


def get_nl_descriptions_for_batch(
    batch: List[MinedResult],
    few_shot_examples: List[AutodocFewShotExample],
    temperature: float = 0.0,
    num_nl_per_query: int = 10,
    max_tokens: int = 64,
    key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> List[Optional[List[str]]]:
    few_shot_c2nl: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl) for ex in few_shot_examples
    ]

    c2nl_engine = code2nl.SimpleCodeToNatLang(
        temperature=temperature, engine=ENGINE, max_tokens=max_tokens
    )
    c2nl_tasks = [
        code2nl.CodeToNatLangTask(
            few_shot_examples=few_shot_c2nl,
            target_code=elem.code,
            task_description="Describe the following data science code snippets in plain english. "
            "Be as exhaustive as possible and repeat any constants verbatim in double quotes. ",
        )
        for elem in batch
    ]

    return c2nl_engine.parallel_get_nl(
        c2nl_tasks,
        num_results=1 if temperature == 0.0 else num_nl_per_query,
        key_manager=key_manager,
    )


def normalize_code_results(
    code_results: List[str], df_arg_names: Set[str]
) -> List[Optional[str]]:
    normalized = []
    for code_result in code_results:
        try:
            #  Sometimes libcst messes up.
            ast.parse(code_result)
            code_result = normalize_code_for_comparison(code_result, df_arg_names)
        except (astlib.cst.ParserSyntaxError, SyntaxError):
            normalized.append(None)
        else:
            normalized.append(code_result)

    return normalized


def evaluate_code_results(
    snippet: MinedResult,
    gt: str,
    candidates: List[str],
    code_results: List[str],
    assistance_level: int,
) -> Tuple[List[AutodocDescription], List[AutodocDescription]]:
    df_arg_names: Set[str] = set(snippet.df_vars)
    correct: List[AutodocDescription] = []
    incorrect: List[AutodocDescription] = []

    code_results = normalize_code_results(code_results, df_arg_names)

    for idx, nl, code_result in zip(range(len(candidates)), candidates, code_results):
        if code_result is None:
            is_equiv = False
            parseable = False
        else:
            parseable = True
            is_equiv = gt == code_result

        success = is_equiv
        desc = AutodocDescription(
            uid=f"{snippet.uid}:{idx}",
            success=success,
            nl=nl,
            generated_code=code_result,
            assistance_level=assistance_level,
            code_parseable=parseable,
        )
        (correct if success else incorrect).append(desc)

    return correct, incorrect


def run_llm_based_parameterization(
    ground_truth_code: str,
    descriptions: List[AutodocDescription],
    few_shot_examples: List[AutodocFewShotExample],
    batch_size: int = 10,
) -> List[Optional[Parameterization]]:
    few_shot_param = [
        FewShotExampleParameterization(
            code=ex.code, nl=ex.nl, param_code=ex.param_code, param_nl=ex.param_nl
        )
        for ex in few_shot_examples
    ]

    results: List[Optional[Parameterization]] = []

    for start_idx in range(0, len(descriptions), batch_size):
        batch = descriptions[start_idx : start_idx + batch_size]

        tasks = [
            ParameterizationTask(
                few_shot_examples=few_shot_param,
                target_nl=desc.nl,
                target_code=desc.generated_code,
                task_description=(
                    "Generalize the code and its natural language description into a reusable function that can be "
                    "applied to other inputs. "
                    "Clearly distinguish which arguments are column parameters. "
                    "Ensure all the arguments are also mentioned in the parameterized natural language."
                ),
            )
            for desc in batch
        ]

        param_engine = SimpleParameterizer(engine=ENGINE)
        for desc, res in zip(batch, param_engine.parallel_parameterize(tasks)):
            if res is None:
                results.append(None)
                continue

            param_nl, param_code = res
            parameterization: Optional[
                Parameterization
            ] = generate_llm_based_parameterization(
                desc,
                param_nl,
                param_code,
                ground_truth_code,
            )
            results.append(parameterization)
            if parameterization:
                print("SUCCESS", parameterization.nl, parameterization.code)
            else:
                print("FAILED FOR", param_nl, param_code)

    return results


def validate_strict(
    snippet: MinedResult,
    gt: str,
    few_shot_examples: List[AutodocFewShotExample],
    candidates: List[str],
    key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> Tuple[List[AutodocDescription], List[AutodocDescription]]:
    few_shot_nl2c: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl) for ex in few_shot_examples
    ]

    assistance_level = 0

    gt_tokens = langmodels.codex_tokenize(snippet.code)["token_ids"]
    nl2c_engine = nl2code.SimpleNatLangToCode(
        temperature=0.0, engine=ENGINE, max_tokens=len(gt_tokens) + 64
    )
    nl2c_tasks = [
        nl2code.NatLangToCodeTask(
            few_shot_examples=few_shot_nl2c,
            target_nl=candidate,
            task_description="Generate a Python pandas code snippet given the english description",
            output_prefix="pd.read_csv(('/kaggle/input/digit-recognizer/'"
            if assistance_level == 1
            else None,
        )
        for candidate in candidates
    ]

    code_results: List[str] = nl2c_engine.parallel_get_code(
        nl2c_tasks,
        key_manager=key_manager,
    )
    desc_correct, desc_incorrect = evaluate_code_results(
        snippet, gt, candidates, code_results, assistance_level
    )

    return desc_correct, desc_incorrect


def perform_corrections(
    gt: str,
    code_result: str,
    nl_candidate: str,
    output_prefix: Optional[str],
    top_logprobs: List[Dict[str, float]],
) -> Optional[str]:
    """Performs corrections and returns the new output prefix, if successful."""
    gt_tok_strs: List[str] = langmodels.codex_tokenize(gt)["token_strs"]
    code_tok_strs: List[str] = langmodels.codex_tokenize(code_result)["token_strs"]

    if output_prefix is not None:
        output_prefix = output_prefix or ""
        output_prefix_tok_strs: List[str] = langmodels.codex_tokenize(output_prefix)[
            "token_strs"
        ]
        top_logprobs = [{tok: 0.0} for tok in output_prefix_tok_strs] + top_logprobs

    new_output_prefix_toks: List[str] = []

    for (gt_tok_idx, gt_tok), code_tok, top_toks in zip(
        enumerate(gt_tok_strs), code_tok_strs, top_logprobs
    ):
        if gt_tok == code_tok:
            new_output_prefix_toks.append(gt_tok)
        elif top_toks is not None and gt_tok in top_toks:
            prefix = gt_tok
            idx = gt_tok_idx
            #  Prevent it from degrading into giving the same result
            while code_tok.startswith(prefix) and idx < len(gt_tok_strs) - 1:
                prefix += gt_tok_strs[idx + 1]
                idx += 1

            new_output_prefix_toks.append(prefix)
            # print("LOOKS GOOD", gt_tok, code_tok, top_toks)
            break
        else:
            #  No repair possible
            return None

    return "".join(new_output_prefix_toks)


def validate_lenient(
    snippet: MinedResult,
    gt: str,
    few_shot_examples: List[AutodocFewShotExample],
    candidates: List[str],
    max_mistakes: int = 2,
    key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> Tuple[List[AutodocDescription], List[AutodocDescription]]:
    few_shot_nl2c: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl) for ex in few_shot_examples
    ]

    #  Candidates still in the running.
    eligible_candidates = candidates[:]
    output_prefixes = [None] * len(eligible_candidates)
    gt_token_ids_and_strs = langmodels.codex_tokenize(snippet.code)
    nl2c_engine = nl2code.SimpleNatLangToCode(
        temperature=0.0,
        engine=ENGINE,
        max_tokens=len(gt_token_ids_and_strs["token_ids"]) + 64,
    )

    correct: List[AutodocDescription] = []
    incorrect: List[AutodocDescription] = []

    for assistance_level in range(1, 1 + max_mistakes + 1):
        nl2c_tasks = [
            nl2code.NatLangToCodeTask(
                few_shot_examples=few_shot_nl2c,
                target_nl=candidate,
                task_description="Generate a Python pandas code snippet given the english description",
                output_prefix=output_prefix,
            )
            for candidate, output_prefix in zip(eligible_candidates, output_prefixes)
        ]

        top_logprobs_list = []
        code_results: List[str] = nl2c_engine.parallel_get_code(
            nl2c_tasks,
            allowed_tokens=gt_token_ids_and_strs["token_ids"],
            key_manager=key_manager,
            top_logprobs=top_logprobs_list,
        )
        desc_correct, desc_incorrect = evaluate_code_results(
            snippet, gt, candidates, code_results, assistance_level
        )

        correct.extend(desc_correct)
        incorrect.extend(desc_incorrect)

        if len(correct) > 0:
            break

        assert len(top_logprobs_list) == len(code_results)
        new_candidates = []
        new_output_prefixes = []
        for candidate, output_prefix, code_result, top_logprobs in zip(
            eligible_candidates, output_prefixes, code_results, top_logprobs_list
        ):
            if top_logprobs is None:
                continue

            new_output_prefix = perform_corrections(
                gt, code_result, candidate, output_prefix, top_logprobs
            )
            if new_output_prefix is not None:
                new_candidates.append(candidate)
                new_output_prefixes.append(new_output_prefix)
            #     print("RETAINED CANDIDATE", candidate)
            #     print("NEW PREFIX", new_output_prefix)
            #     print(gt, code_result)
            # else:
            #     print("FAILED TO RETAIN CANDIDATE", candidate)
            #     print(gt, code_result)

        eligible_candidates = new_candidates
        output_prefixes = new_output_prefixes
        if len(eligible_candidates) == 0:
            break

    return correct, incorrect


def validate_nl_descriptions(
    snippet: MinedResult,
    few_shot_examples: List[AutodocFewShotExample],
    candidates: List[str],
    max_mistakes: int = 1,
    key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> AutodocResult:
    df_arg_names: Set[str] = set(snippet.df_vars)
    gt = normalize_code_for_comparison(snippet.code, df_arg_names)

    correct: List[AutodocDescription] = []
    incorrect: List[AutodocDescription] = []

    #  First try the strongest, validate_strict
    desc_correct, desc_incorrect = validate_strict(
        snippet, gt, few_shot_examples, candidates, key_manager
    )
    correct.extend(desc_correct)
    incorrect.extend(desc_incorrect)

    if len(correct) == 0:
        #  Try a more lenient validator
        desc_correct, desc_incorrect = validate_lenient(
            snippet, gt, few_shot_examples, candidates, max_mistakes, key_manager
        )
        correct.extend(desc_correct)
        incorrect.extend(desc_incorrect)

    if len(correct) > 0:
        llm_based_parameterizations = run_llm_based_parameterization(
            gt, correct, few_shot_examples
        )
        for idx, desc in enumerate(correct):
            desc.llm_based_parameterization = llm_based_parameterizations[idx]
            try:
                desc.nl_based_parameterization = generate_nl_based_parameterization(
                    snippet, desc.nl
                )
            except:
                pass

    print(f"Code: {gt}")
    print("Correct:")
    for desc in sorted(correct, key=lambda x: x.assistance_level):
        print(
            f"[{desc.assistance_level}] NL: {desc.nl} || Generated Code: {desc.generated_code}"
        )
        if desc.llm_based_parameterization is not None:
            print(f"[LLM] Param NL: {desc.llm_based_parameterization.nl}")
            print(f"[LLM] Param Code:\n{desc.llm_based_parameterization.code}")

        if desc.nl_based_parameterization is not None:
            print(f"[NLBased] Param NL: {desc.nl_based_parameterization.nl}")
            print(f"[NLBased] Param Code:\n{desc.nl_based_parameterization.code}")

        print("-----")

    print("---")
    print("Incorrect:")
    for desc in sorted(incorrect, key=lambda x: x.assistance_level):
        print(
            f"[{desc.assistance_level}] NL: {desc.nl} || Generated Code: {desc.generated_code}"
        )
    print("---")

    return AutodocResult(
        uid=snippet.uid,
        success=len(correct) > 0,
        ground_truth_code=gt,
        correct_descriptions=correct,
        incorrect_descriptions=incorrect,
    )


def run_autodoc_for_batch(
    batch: List[MinedResult],
    few_shot_examples: List[AutodocFewShotExample],
    temperature: float = 0.0,
    num_nl_per_query: int = 10,
) -> List[AutodocResult]:
    available_keys = langmodels.get_available_keys()

    c2nl_key_manager = langmodels.OpenAIKeyManager(keys=available_keys)
    nl2c_key_manager = langmodels.OpenAIKeyManager(keys=available_keys)
    #
    # batch[0].code = "round(df['Price'], 2)"
    # batch[0].df_vars = ['df']
    # batch[0].code = "pd.merge(df1, df2, on=['shop_id'], how='inner').fillna(0)"
    # batch[0].df_vars = ["df1", "df2"]

    #  Get NL for each in one shot using parallel prompts
    nl_descriptions = get_nl_descriptions_for_batch(
        batch,
        few_shot_examples,
        temperature,
        num_nl_per_query,
        key_manager=c2nl_key_manager,
    )
    num_success = 0
    autodoc_results: List[AutodocResult] = []
    for desc_candidates, snippet in zip(nl_descriptions, batch):
        print("Code:", snippet.code)
        for k in desc_candidates:
            print("*", k)

        print("-------")
        autodoc_res = validate_nl_descriptions(
            snippet, few_shot_examples, desc_candidates, key_manager=nl2c_key_manager
        )
        print("=======")
        if autodoc_res.success:
            num_success += 1

        autodoc_results.append(autodoc_res)

    return autodoc_results


def try_transferring_autodoc_result(
    orig_snippet: MinedResult, new_snippet: MinedResult, autodoc_result: AutodocResult
) -> List[AutodocDescription]:
    assert autodoc_result.success
    transferred_correct_descs: List[AutodocDescription] = []

    for desc in autodoc_result.correct_descriptions:
        if (
            desc.llm_based_parameterization is None
            and desc.nl_based_parameterization is None
        ):
            continue

        transferred_nl: Optional[str] = None
        new_llm_parameterization: Optional[Parameterization] = None
        new_nl_parameterization: Optional[Parameterization] = None

        if desc.nl_based_parameterization is not None:
            new_nl_parameterization = attempt_parameterization_application(
                desc.nl_based_parameterization, new_snippet
            )
            if new_nl_parameterization is not None:
                transferred_nl = new_nl_parameterization.get_instantiated_nl()

        if desc.llm_based_parameterization is not None:
            new_llm_parameterization = attempt_parameterization_application(
                desc.llm_based_parameterization, new_snippet
            )
            if new_llm_parameterization:
                transferred_nl = new_llm_parameterization.get_instantiated_nl()

        if (new_llm_parameterization is not None) or (
            new_nl_parameterization is not None
        ):
            assert transferred_nl is not None
            new_desc = AutodocDescription(
                uid=desc.uid,
                success=True,
                nl=transferred_nl,
                generated_code=new_snippet.code,
                code_parseable=True,
                assistance_level=desc.assistance_level,
                llm_based_parameterization=new_llm_parameterization,
                nl_based_parameterization=new_nl_parameterization,
                is_derived=True,
            )
            transferred_correct_descs.append(new_desc)
            # print("\n=====\nTRANSFER SUCCESSFUL FOR", desc.nl)
            # print(new_snippet.code)
            # if new_llm_parameterization is not None:
            #     print("[LLM] INSTANTIATED NL", new_llm_parameterization.get_instantiated_nl())
            # if new_nl_parameterization is not None:
            #     print("[NL] INSTANTIATED NL", new_nl_parameterization.get_instantiated_nl())
        # else:
        #     print("FAILED FOR", desc.nl)

    return transferred_correct_descs


def run_preprocessing(campaign_dir: str, rerun_preprocessing: bool = False) -> None:
    """Prepare campaign for autodoc"""
    #  Load the code and template for each mining result into memory. This will be used to share autodoc results
    #  among multiple snippets.
    preprocessing_path = os.path.join(campaign_dir, PREPROCESSING_RESULTS_FILE)
    if os.path.exists(preprocessing_path) and not rerun_preprocessing:
        return

    mining_results_path = os.path.join(campaign_dir, MINING_RESULTS_FILE)
    if not os.path.exists(mining_results_path):
        raise FileNotFoundError(
            f"Could not find mining results at {mining_results_path}"
        )

    simplified_entries: List[Dict] = []
    with pickleutils.PickledMapReader(mining_results_path) as reader:
        print(f"Found {len(reader)} mining results")
        for idx, mining_result in enumerate(
            tqdm.tqdm(
                reader.values(),
                desc="Running Preprocessing",
                dynamic_ncols=True,
                total=len(reader),
            )
        ):
            assert isinstance(mining_result, MinedResult)
            simplified_entries.append(
                {
                    "uid": mining_result.uid,
                    "code": mining_result.code,
                    "template": mining_result.template,
                    "support": 1,
                }
            )

    #  Deduplicate by code
    code_counter = collections.Counter(x["code"] for x in simplified_entries)
    simplified_entries = list(
        {
            entry["code"]: {**entry, "support": code_counter[entry["code"]]}
            for entry in simplified_entries
        }.values()
    )
    print(f"Found {len(simplified_entries)} code-unique mining results")

    #  Save to disk
    pickleutils.smart_dump(simplified_entries, preprocessing_path)


def compute_template_processing_order(preprocessed_entries: List[Dict]) -> List[str]:
    template_supports: Dict[str, int] = collections.Counter()
    for entry in preprocessed_entries:
        template_supports[entry["template"]] += entry["support"]

    sorted_items = sorted(template_supports.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_items]


def build_next_chunk(
    active_iters: List[Tuple[str, Iterator[Dict]]],
    template_iters_queue: Deque[Tuple[str, Iterator[Dict]]],
    finished_uids: Set[str],
) -> List[Dict]:
    chunk: List[Dict] = []
    iter_worklist: Deque[Tuple[str, Iterator[Dict]]] = collections.deque(active_iters)
    new_active_iters: List[Tuple[str, Iterator[Dict]]] = []

    while len(iter_worklist) > 0:
        template, iter_ = iter_worklist.popleft()
        for entry in iter_:
            if entry["uid"] not in finished_uids:
                chunk.append(entry)
                new_active_iters.append((template, iter_))
                break
        else:
            if len(template_iters_queue) > 0:
                iter_worklist.append(template_iters_queue.popleft())

    active_iters.clear()
    active_iters.extend(new_active_iters)

    return chunk


def run_autodoc_new(
    campaign_dir: str,
    few_shot_version: int = 1,
    batch_size: int = 10,
    rerun_preprocessing: bool = False,
) -> None:
    """Run autodoc for a campaign assuming the few-shot examples have been set up."""
    run_preprocessing(campaign_dir, rerun_preprocessing)

    preprocessing_path = os.path.join(campaign_dir, PREPROCESSING_RESULTS_FILE)
    simplified_entries: List[Dict] = pickleutils.smart_load(preprocessing_path)

    print(f"Found {len(simplified_entries)} entries")

    mining_results_path = os.path.join(campaign_dir, MINING_RESULTS_FILE)
    if not os.path.exists(mining_results_path):
        raise FileNotFoundError(
            f"Could not find mining results at {mining_results_path}"
        )

    #  Load few shot examples
    few_shot_path = get_few_shot_example_path(campaign_dir, few_shot_version)
    if not os.path.exists(few_shot_path):
        raise FileNotFoundError(f"Could not find few shot examples at {few_shot_path}")

    with open(few_shot_path, "r") as f:
        few_shot_dicts: List[Dict] = yaml.full_load(f)
        if any("TODO" in ex["nl"] for ex in few_shot_dicts):
            raise ValueError(f"Some few-shot examples are missing an NL description")

    few_shot_examples = [AutodocFewShotExample.from_json(ex) for ex in few_shot_dicts]

    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    autodoc_failures_path = os.path.join(campaign_dir, AUTODOC_FAILURES_FILE)

    #  Track the UIDs that have been processed (regardless of whether it was successful or not)
    finished: Set[str] = set()
    #  Track successful and unsuccessful UIDs
    successful_uids: Set[str] = set()
    unsuccessful_uids: Set[str] = set()

    #  Load existing results if any
    if os.path.exists(autodoc_successes_path):
        with pickleutils.PickledMapReader(autodoc_successes_path) as reader:
            finished.update(reader.keys())
            successful_uids.update(reader.keys())

    if os.path.exists(autodoc_failures_path):
        with pickleutils.PickledMapReader(autodoc_failures_path) as reader:
            finished.update(reader.keys())
            unsuccessful_uids.update(reader.keys())

    if len(finished) > 0:
        print(f"Already processed {len(finished)} entries")
        print(f"Already found {len(unsuccessful_uids)} unsuccessful autodoc results")
        print(f"Already found {len(successful_uids)} successful autodoc results")

    seen_templates: Set[str] = {
        e["template"] for e in simplified_entries if e["uid"] in finished
    }
    simplified_entries = [e for e in simplified_entries if e["uid"] not in finished]

    #  Collect entries for each template
    entries_by_template: Dict[str, List[Dict]] = collections.defaultdict(list)
    for entry in simplified_entries:
        entries_by_template[entry["template"]].append(entry)

    #  Shuffle the entries for each template
    for template, entries in entries_by_template.items():
        random.shuffle(entries)

    template_order: List[str] = compute_template_processing_order(simplified_entries)
    template_iters_queue: Deque[Tuple[str, Iterator[Dict]]] = collections.deque()
    for template in template_order:
        template_iters_queue.append((template, iter(entries_by_template[template])))

    #  Maintain a queue of active iterators that we'll use for building chunks
    active_iters: List[Tuple[str, Iterator[Dict]]] = []
    while (len(active_iters) < batch_size) and (len(template_iters_queue) > 0):
        active_iters.append(template_iters_queue.popleft())

    #  We'll also maintain a counter of how many times a template continuously failed.
    #  This will help us demote a template to the back if it's failing too much.
    template_failure_counts: Dict[str, int] = collections.defaultdict(int)
    with pickleutils.PickledMapReader(
        mining_results_path
    ) as mined_result_reader, tqdm.tqdm(
        desc="Running Autodoc", dynamic_ncols=True, total=len(simplified_entries)
    ) as pbar, pickleutils.PickledMapWriter(
        autodoc_successes_path, overwrite_existing=False
    ) as writer_success, pickleutils.PickledMapWriter(
        autodoc_failures_path, overwrite_existing=False
    ) as writer_failures:

        while len(finished) < len(finished) + len(simplified_entries):
            num_finished_start = len(finished)

            #  Build the next chunk of entries to process
            chunk: List[Dict] = build_next_chunk(
                active_iters, template_iters_queue, finished
            )
            batch: List[MinedResult] = [mined_result_reader[x["uid"]] for x in chunk]

            #  Run autodoc for the batch
            autodoc_results = run_autodoc_for_batch(
                batch, few_shot_examples, temperature=0.8, num_nl_per_query=10
            )
            assert len(autodoc_results) == len(batch) == len(chunk)

            for entry, (snippet_idx, snippet), autodoc_res in zip(
                chunk, enumerate(batch), autodoc_results
            ):
                finished.add(entry["uid"])
                success = autodoc_res.success
                (successful_uids if success else unsuccessful_uids).add(entry["uid"])
                if not success:
                    writer_failures[entry["uid"]] = autodoc_res
                    template_failure_counts[entry["template"]] += 1
                    continue
                else:
                    writer_success[entry["uid"]] = autodoc_res
                    seen_templates.add(entry["template"])
                    #  Try transferring the result to other results with the same template.
                    unprocessed_entries = [
                        e
                        for e in entries_by_template[entry["template"]]
                        if e["uid"] not in finished
                    ]
                    with tqdm.tqdm(
                        unprocessed_entries,
                        desc=f"Transferring results for {snippet_idx}",
                        dynamic_ncols=True,
                    ) as transfer_pbar:
                        transfer_succ, transfer_fail = 0, 0
                        for unprocessed_entry in transfer_pbar:
                            unprocessed_snippet = mined_result_reader[
                                unprocessed_entry["uid"]
                            ]
                            assert isinstance(unprocessed_snippet, MinedResult)
                            transferred_descs = try_transferring_autodoc_result(
                                snippet, unprocessed_snippet, autodoc_res
                            )
                            if len(transferred_descs) > 0:
                                transferred_res = AutodocResult(
                                    uid=unprocessed_entry["uid"],
                                    success=True,
                                    ground_truth_code=unprocessed_snippet.code,
                                    correct_descriptions=transferred_descs,
                                    incorrect_descriptions=[],
                                )
                                writer_success[
                                    unprocessed_entry["uid"]
                                ] = transferred_res
                                finished.add(unprocessed_entry["uid"])
                                successful_uids.add(unprocessed_entry["uid"])
                                transfer_succ += 1
                            else:
                                transfer_fail += 1

                            transfer_pbar.set_postfix(
                                succ=transfer_succ, fail=transfer_fail
                            )

                #  Reset the contiguous failure count
                template_failure_counts[entry["template"]] = 0

            pbar.update(len(finished) - num_finished_start)
            pbar.set_postfix(
                successes=len(successful_uids),
                failures=len(unsuccessful_uids),
                templates=len(seen_templates),
            )

            #  Demote a template if too many errors
            for idx, (template, iter_) in enumerate(active_iters, start=0):
                if (
                    template_failure_counts[template] >= MAX_TEMPLATE_FAILURES
                    and len(template_iters_queue) > 0
                ):
                    active_iters[idx] = template_iters_queue.popleft()
                    template_iters_queue.append((template, iter_))

            writer_success.flush()
            writer_failures.flush()


def prepare_dataset_for_generational_model(
    campaign_dir: str, num_per_desc_uid: int = 10
):
    """Prepare a dataset out of the autodoc results to be used for training a small model like CodeBERT or CodeT5"""
    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    print(f"Found {len(all_autodoc_results)} autodoc results")
    descriptions_by_uid: Dict[str, List[AutodocDescription]] = collections.defaultdict(
        list
    )
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            descriptions_by_uid[desc.uid].append(desc)

    #  Shuffle each array
    for descs in descriptions_by_uid.values():
        random.shuffle(descs)

    print(f"Found {len(descriptions_by_uid)} unique descriptions")

    iter_dict: Dict[str, Iterator[AutodocDescription]] = {
        k: itertools.cycle(v) for k, v in descriptions_by_uid.items()
    }

    records = []
    for loop_no in range(0, num_per_desc_uid):
        for iter_ in iter_dict.values():
            desc = next(iter_)
            nl = desc.nl
            code = desc.generated_code
            if code.startswith("def "):
                code = "\n".join(code.split("\n")[1:])
            if code.startswith("return "):
                code = code[len("return ") :]

            source_text = f"generate-code: {nl}"
            target_text = code

            records.append({"source_text": source_text, "target_text": target_text})

    df = pd.DataFrame.from_records(records)
    print(df)
    pickleutils.smart_dump(df, os.path.join(campaign_dir, "t5_code_data.pkl"))


def get_max_tokens(tokenizer, strings: List[str]) -> int:
    lengths = []
    for idx in range(0, len(strings), 32):
        batch = strings[idx : idx + 32]
        lengths.extend(len(i) for i in tokenizer(batch)["input_ids"])

    print(
        np.mean(lengths),
        np.median(lengths),
        np.max(lengths),
        np.min(lengths),
        np.percentile(lengths, 50),
        np.percentile(lengths, 75),
        np.percentile(lengths, 90),
        np.percentile(lengths, 99),
    )
    return max(lengths)


def train_generational_model(campaign_dir: str, model_name: str, max_epochs: int = 10):
    df = pickleutils.smart_load(os.path.join(campaign_dir, "t5_code_data.pkl"))
    train_df, test_df = train_test_split(df, test_size=0.2)

    if model_name.startswith("Salesforce/codet5"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    simplet5_model = SimpleT5()
    simplet5_model.tokenizer = tokenizer
    simplet5_model.model = model

    prefix = model_name.replace("/", "_")

    output_dir = os.path.join(campaign_dir, prefix + "_model_outputs")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    simplet5_model.train(
        train_df=train_df,
        eval_df=test_df,
        source_max_token_len=128,
        target_max_token_len=128,
        outputdir=output_dir,
        early_stopping_patience_epochs=3,
        batch_size=8,
        max_epochs=max_epochs,
        use_gpu=True,
    )

    simplet5_model.model.save_pretrained(
        os.path.join(campaign_dir, prefix + "_trained")
    )
    simplet5_model.tokenizer.save_pretrained(
        os.path.join(campaign_dir, prefix + "_trained")
    )


def prepare_dataset_for_training_embeddings(
    campaign_dir: str, per_equiv_class: int = 10
):
    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    desc_by_uids: Dict[str, List[AutodocDescription]] = collections.defaultdict(list)
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            if desc.is_derived:
                desc_by_uids[desc.uid].append(desc)

    non_derived_results: List[AutodocResult] = [
        res
        for res in all_autodoc_results
        if all(not desc.is_derived for desc in res.correct_descriptions)
    ]
    print(f"Found {len(non_derived_results)} non-derived autodoc results")

    equiv_classes: List[List[str]] = []
    for res in non_derived_results:
        unique_nls: Set[str] = set()
        for desc in res.correct_descriptions:
            unique_nls.add(desc.nl)
            if desc.llm_based_parameterization is not None:
                unique_nls.add(desc.llm_based_parameterization.nl)

            if len(desc_by_uids[desc.uid]) >= 1:
                unique_nls.add(random.choice(desc_by_uids[desc.uid]).nl)

        unique_nls.update(desc.nl for desc in res.incorrect_descriptions)
        equiv_classes.append(list(unique_nls))
        random.shuffle(equiv_classes[-1])

    all_indices: List[int] = list(range(len(equiv_classes)))
    correct_pairs: List[Tuple[str, str]] = []
    incorrect_pairs: List[Tuple[str, str]] = []
    for idx, equiv_class in enumerate(tqdm.tqdm(equiv_classes, desc="Preparing data")):
        candidates = list(
            itertools.islice(itertools.combinations(equiv_class, 2), per_equiv_class)
        )
        random.shuffle(candidates)

        for s1, s2 in candidates:
            correct_pairs.append((s1, s2))

        other_sample = random.sample(all_indices, per_equiv_class)
        while idx in other_sample:
            other_sample = random.sample(all_indices, per_equiv_class)

        for other_idx in other_sample:
            s1 = random.choice(equiv_class)
            s2 = random.choice(equiv_classes[other_idx])
            incorrect_pairs.append((s1, s2))

    print("SIZES", len(correct_pairs), len(incorrect_pairs))
    pickleutils.smart_dump(
        [correct_pairs, incorrect_pairs],
        os.path.join(campaign_dir, "embedding_train_data.pkl"),
    )


def train_embeddings(
    campaign_dir: str, loss_type: str = "contrastive", num_epochs: int = 5
):
    correct_pairs, incorrect_pairs = pickleutils.smart_load(
        os.path.join(campaign_dir, "embedding_train_data.pkl")
    )
    random.shuffle(correct_pairs)
    random.shuffle(incorrect_pairs)

    model = SentenceTransformer("all-mpnet-base-v2")
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
        output_path = os.path.join(campaign_dir, "embedding_model_contrastive")
    elif loss_type == "cosine":
        train_loss = losses.CosineSimilarityLoss(model=model)
        output_path = os.path.join(campaign_dir, "embedding_model_cosine")
        for ex in itertools.chain(train_examples, test_examples):
            ex.label = float(ex.label)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_examples, show_progress_bar=True
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    model.fit(
        [(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=10000,
        show_progress_bar=True,
        save_best_model=True,
        output_path=output_path,
    )


def prepare_search_engine(campaign_dir: str, model_path: str):
    model = SentenceTransformer(model_path)

    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    corpus: List[AutodocDescription] = []
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            if desc.is_derived:
                continue
            corpus.append(desc)

    print(f"Found {len(corpus)} autodoc descriptions")

    embeddings = model.encode(
        [desc.nl for desc in corpus], show_progress_bar=True, convert_to_tensor=True
    )
    print("Finished generating embeddings")

    embeddings_path = os.path.join(model_path, "search_engine_embeddings.pkl")
    pickleutils.smart_dump((corpus, embeddings), embeddings_path)
    print(f"Saved corpus and embeddings to {embeddings_path}")


def start_search_engine(campaign_dir: str, model_path: str):
    corpus, embeddings = pickleutils.smart_load(
        os.path.join(model_path, "search_engine_embeddings.pkl")
    )
    print(f"Loaded corpus and embeddings from {model_path}")

    model = SentenceTransformer(model_path)
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
            if corpus[idx].generated_code in seen_code:
                continue

            seen_code.add(corpus[idx].generated_code)
            print(corpus[idx].nl, "(Score: {:.4f})".format(score))
            print(corpus[idx].generated_code)
            print("---")
            ctr += 1

            if ctr == 10:
                break

        print("\n----------\n")


def run_search_engine(model_path: str, queries: List[str]):
    corpus, embeddings = pickleutils.smart_load(
        os.path.join(model_path, "search_engine_embeddings.pkl")
    )
    print(f"Loaded corpus and embeddings from {model_path}")

    model = SentenceTransformer(model_path)
    all_results: List[List[Dict]] = []
    for query in tqdm.tqdm(queries, desc="Processing queries"):
        query_embedding = model.encode(
            query, show_progress_bar=False, convert_to_tensor=True
        )
        distances = util.cos_sim(query_embedding, embeddings)[0]

        top_results = torch.topk(distances, k=100)
        seen_code: Set[str] = set()
        ctr = 0
        results: List[Dict] = []
        for score, idx in zip(top_results[0], top_results[1]):
            if corpus[idx].generated_code in seen_code:
                continue

            seen_code.add(corpus[idx].generated_code)
            results.append(
                {
                    "score": score,
                    "nl": corpus[idx].nl,
                    "code": corpus[idx].generated_code,
                }
            )
            ctr += 1

            if ctr == 10:
                break

        all_results.append(results)

    return all_results


def analyze(campaign_dir: str):
    autodoc_results_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    autodoc_failures_path = os.path.join(campaign_dir, AUTODOC_FAILURES_FILE)

    with pickleutils.PickledMapReader(autodoc_results_path) as reader:
        for uid, res in reader.items():
            assert isinstance(res, AutodocResult)
            print("UID", uid, res.success)
            print("CODE:", res.ground_truth_code)
            for desc in res.correct_descriptions:
                print(
                    desc.nl,
                    " ||| ",
                    desc.parameterization.nl,
                    "|||",
                    desc.parameterization.code,
                )

            print("---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    fire.Fire(
        {
            "run_autodoc": run_autodoc_new,
            "analyze": analyze,
            "prepare_dataset_for_generational_model": prepare_dataset_for_generational_model,
            "train_generational_model": train_generational_model,
            "prepare_dataset_for_training_embeddings": prepare_dataset_for_training_embeddings,
            "train_embeddings": train_embeddings,
            "prepare_search_engine": prepare_search_engine,
            "start_search_engine": start_search_engine,
        }
    )
