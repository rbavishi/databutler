import collections
import itertools
import os
import random
from typing import Optional, List, Dict, Tuple, Set

import fire
import tqdm
import yaml

from databutler.datana.generic.autodoc import code2nl, nl2code
from databutler.datana.generic.autodoc.few_shot import FewShotExampleCodeAndNL
from databutler.mining.kaggle.static_analysis.pandas_autodoc_utils import AutodocFewShotExample, \
    AutodocDescription, AutodocResult, normalize_code_for_comparison, get_few_shot_example_path, \
    parameterize_snippet, apply_parameterization
from databutler.mining.kaggle.static_analysis.pandas_mining import MINING_RESULTS_FILE
from databutler.mining.kaggle.static_analysis.pandas_mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import pickleutils, langmodels

ENGINE = 'code-davinci-002'
PREPROCESSING_RESULTS_FILE = "pandas_mining_preprocessed.pkl"
AUTODOC_RESULTS_FILE = "autodoc_results.pkl"
AUTODOC_FAILURES_FILE = "autodoc_failures.pkl"


def get_nl_descriptions_for_batch(
        batch: List[MinedResult],
        few_shot_examples: List[AutodocFewShotExample],
        temperature: float = 0.0,
        num_nl_per_query: int = 10,
        max_tokens: int = 64,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> List[Optional[List[str]]]:
    few_shot_c2nl: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl)
        for ex in few_shot_examples
    ]

    c2nl_engine = code2nl.SimpleCodeToNatLang(temperature=temperature, engine=ENGINE, max_tokens=max_tokens)
    c2nl_tasks = [
        code2nl.CodeToNatLangTask(
            few_shot_examples=few_shot_c2nl,
            target_code=elem.code,
            task_description="Describe the following data science code snippets in plain english. "
                             "Be as exhaustive as possible and repeat any constants verbatim in double quotes. "
                             "Clearly indicate which values are columns using the COL: prefix."
        )
        for elem in batch
    ]

    return c2nl_engine.parallel_get_nl(c2nl_tasks, num_results=1 if temperature == 0.0 else num_nl_per_query,
                                       key_manager=key_manager)


def validate_nl_descriptions(
        snippet: MinedResult,
        few_shot_examples: List[AutodocFewShotExample],
        candidates: List[str],
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> AutodocResult:
    few_shot_nl2c: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl)
        for ex in few_shot_examples
    ]

    df_arg_names: Set[str] = set(snippet.df_vars)
    gt = normalize_code_for_comparison(snippet.code, df_arg_names)

    gt_tokens = langmodels.codex_tokenize(snippet.code)["token_ids"]
    nl2c_engine = nl2code.SimpleNatLangToCode(temperature=0.0, engine=ENGINE, max_tokens=len(gt_tokens) + 64)

    correct: List[AutodocDescription] = []
    incorrect: List[AutodocDescription] = []
    for allowed_tokens, assistance_level in [(None, 0), (gt_tokens, 1)]:
        nl2c_tasks = [
            nl2code.NatLangToCodeTask(
                few_shot_examples=few_shot_nl2c,
                target_nl=candidate,
                task_description="Generate a Python pandas code snippet given the english description",
            ) for candidate in candidates
        ]

        code_results: List[str] = nl2c_engine.parallel_get_code(
            nl2c_tasks,
            allowed_tokens=allowed_tokens,
            key_manager=key_manager,
        )
        for idx, nl, code_result in zip(range(len(candidates)), candidates, code_results):
            try:
                code_result = normalize_code_for_comparison(code_result, df_arg_names)
            except (astlib.cst.ParserSyntaxError, SyntaxError):
                is_equiv = False
                parseable = False
                parameterization = None
            else:
                parseable = True
                is_equiv = gt == code_result
                parameterization = None
                try:
                    if is_equiv:
                        parameterization = parameterize_snippet(snippet, nl)
                except:
                    pass

            success = is_equiv and parameterization is not None
            desc = AutodocDescription(
                uid=f"{snippet.uid}:{idx}",
                success=success,
                nl=nl,
                generated_code=code_result,
                assistance_level=assistance_level,
                code_parseable=parseable,
                parameterization=parameterization,
            )
            (correct if success else incorrect).append(desc)

        if len(correct) > 0:
            break

    print(f"Code: {gt}")
    print("Correct:")
    for desc in correct:
        print(f"NL: {desc.nl} || Generated Code: {desc.generated_code}")
    print("---")
    print("Incorrect:")
    for desc in incorrect:
        print(f"NL: {desc.nl} || Generated Code: {desc.generated_code}")
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
    if len(available_keys) < 2:
        print("WARNING: At least two keys are recommended")
        available_keys = available_keys * 2

    c2nl_key_manager = langmodels.OpenAIKeyManager(keys=available_keys)
    nl2c_key_manager = langmodels.OpenAIKeyManager(keys=available_keys)

    #  Get NL for each in one shot using parallel prompts
    nl_descriptions = get_nl_descriptions_for_batch(
        batch, few_shot_examples, temperature, num_nl_per_query, key_manager=c2nl_key_manager
    )
    num_success = 0
    autodoc_results: List[AutodocResult] = []
    for desc_candidates, snippet in zip(nl_descriptions, batch):
        print("Code:", snippet.code)
        for k in desc_candidates:
            print("*", k)

        print("-------")
        autodoc_res = validate_nl_descriptions(snippet, few_shot_examples, desc_candidates,
                                               key_manager=nl2c_key_manager)
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
        assert desc.parameterization is not None
        new_parameterization = apply_parameterization(desc.parameterization, new_snippet)
        if new_parameterization is not None:
            transferred_correct_descs.append(
                AutodocDescription(
                    uid=desc.uid,
                    success=True,
                    nl=desc.nl,
                    generated_code=desc.generated_code,
                    code_parseable=desc.code_parseable,
                    assistance_level=desc.assistance_level,
                    parameterization=new_parameterization,
                    is_derived=True,
                )
            )
            # print("GOT", new_snippet.code, new_parameterization.instantiation)
        # else:
        #     print("FAILED FOR", new_snippet.code, "FROM", orig_snippet.code)
        pass

    return transferred_correct_descs


def run_autodoc(
        campaign_dir: str,
        few_shot_version: int = 1,
        batch_size: int = 10,
        num_results: Optional[int] = None,
        retry_failures: bool = False,
) -> None:
    """Run autodoc for a campaign assuming the few-shot examples have been set up."""
    mining_results_path = os.path.join(campaign_dir, MINING_RESULTS_FILE)
    if not os.path.exists(mining_results_path):
        raise FileNotFoundError(f"Could not find mining results at {mining_results_path}")

    #  Load the code and template for each mining result into memory. This will be used to share autodoc results
    #  among multiple snippets.
    preprocessing_path = os.path.join(campaign_dir, PREPROCESSING_RESULTS_FILE)
    if os.path.exists(preprocessing_path):
        uids_to_process, templates_to_snippet_dict = pickleutils.smart_load(preprocessing_path)
    else:
        templates_to_snippet_dict: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
        uids_to_process: List[str] = []
        with pickleutils.PickledMapReader(mining_results_path) as reader:
            print(f"Found {len(reader)} mining results")
            #  NOTE: This is assuming that we can fit everything in memory. For a few million results,
            #  this should be okay (max 20 GB).
            if num_results is None or len(reader) <= num_results:
                iterator = reader.values()
                length = len(reader)
            else:
                print(f"Only considering {num_results} results at max")
                iterator = itertools.islice(reader.values(), num_results)
                length = num_results

            for res in tqdm.tqdm(iterator, total=length, dynamic_ncols=True, desc="Preprocessing results"):
                uids_to_process.append(res.uid)
                templates_to_snippet_dict[res.template].append((res.uid, res.code))

        pickleutils.smart_dump((uids_to_process, templates_to_snippet_dict), preprocessing_path)

    print(f"Found {len(uids_to_process)} mining results")
    print(f"Found {len(templates_to_snippet_dict)} unique templates")

    #  Load few shot examples
    few_shot_path = get_few_shot_example_path(campaign_dir, few_shot_version)
    if not os.path.exists(few_shot_path):
        raise FileNotFoundError(f"Could not find few shot examples at {few_shot_path}")

    with open(few_shot_path, 'r') as f:
        few_shot_dicts: List[Dict] = yaml.full_load(f)
        if any("TODO" in ex["nl"] for ex in few_shot_dicts):
            raise ValueError(f"Some few-shot examples are missing an NL description")

    few_shot_examples = [AutodocFewShotExample.from_json(ex) for ex in few_shot_dicts]

    success_uids: Set[str] = set()
    failure_uids: Set[str] = set()
    autodoc_results_path = os.path.join(campaign_dir, AUTODOC_RESULTS_FILE)
    autodoc_failures_path = os.path.join(campaign_dir, AUTODOC_FAILURES_FILE)

    with pickleutils.PickledMapReader(mining_results_path) as reader, \
            pickleutils.PickledMapWriter(autodoc_results_path, overwrite_existing=False) as writer_success, \
            pickleutils.PickledMapWriter(autodoc_failures_path, overwrite_existing=False) as writer_failures:
        already_processed: Set[str] = set(writer_success.keys())
        success_uids.update(already_processed)
        if not retry_failures:
            failure_uids.update(writer_failures.keys())
            already_processed.update(failure_uids)

        if len(already_processed) > 0:
            print(f"Already processed {len(already_processed)} snippets. "
                  f"Delete file at {autodoc_results_path} and/or {autodoc_failures_path} to reset.")
        uid_processing_order: List[str] = []
        for template, values in sorted(templates_to_snippet_dict.items(), key=lambda x: -len(x[1])):
            uid_processing_order.extend(uid for uid, _ in sorted(values, key=lambda x: len(x[1]))
                                        if uid not in already_processed)
        # uids_to_process = [uid for uid in uid_processing_order
        #                    if len(templates_to_snippet_dict[reader[uid].template]) >= 3]
        uids_to_process = uid_processing_order

        def _update_stats(_pbar):
            _pbar.set_postfix(success=len(success_uids), failures=len(failure_uids))

        # random.shuffle(uids_to_process)
        chunk: List[MinedResult] = []
        with tqdm.tqdm(uids_to_process, dynamic_ncols=True, desc="Running Autodoc") as pbar:
            for uid in pbar:
                if uid in success_uids:
                    _update_stats(pbar)
                    continue

                chunk.append(reader[uid])

                if len(chunk) < batch_size and uid != uids_to_process[-1]:
                    _update_stats(pbar)
                    continue
                elif len(chunk) > 0:
                    autodoc_results = run_autodoc_for_batch(chunk, few_shot_examples,
                                                            temperature=0.8, num_nl_per_query=10)
                    chunk_uids: Set[str] = {snippet.uid for snippet in chunk}
                    new_autodoc_descriptions: Dict[str, List[AutodocDescription]] = collections.defaultdict(list)
                    for snippet, autodoc_result in zip(chunk, autodoc_results):
                        if not autodoc_result.success:
                            failure_uids.add(snippet.uid)
                            writer_failures[snippet.uid] = autodoc_result
                            continue

                        success_uids.add(snippet.uid)
                        writer_success[snippet.uid] = autodoc_result

                        todo_snippets: List[MinedResult] = [
                            reader[u] for (u, _) in templates_to_snippet_dict[snippet.template]
                            if u not in chunk_uids and u not in success_uids
                        ]

                        if len(todo_snippets) == 0:
                            continue

                        #  Try to reuse the results if possible.
                        for todo_snippet in todo_snippets:
                            descs = try_transferring_autodoc_result(snippet, todo_snippet, autodoc_result)
                            if len(descs) > 0:
                                new_autodoc_descriptions[todo_snippet.uid].extend(descs)

                    #  Collect the ones where it actually worked
                    for todo_uid, descs in new_autodoc_descriptions.items():
                        if len(descs) > 0:
                            success_uids.add(todo_uid)
                            failure_uids.discard(todo_uid)
                            writer_success[todo_uid] = AutodocResult(
                                uid=todo_uid,
                                success=True,
                                ground_truth_code=reader[todo_uid].code,
                                correct_descriptions=descs,
                                incorrect_descriptions=[],
                            )
                            print("ALSO SATISFIED", todo_uid)

                    chunk.clear()
                    writer_success.flush()
                    writer_failures.flush()
                    _update_stats(pbar)


def analyze(campaign_dir: str):
    autodoc_results_path = os.path.join(campaign_dir, AUTODOC_RESULTS_FILE)
    autodoc_failures_path = os.path.join(campaign_dir, AUTODOC_FAILURES_FILE)

    with pickleutils.PickledMapReader(autodoc_results_path) as reader:
        for uid, res in reader.items():
            assert isinstance(res, AutodocResult)
            print("UID", uid, res.success)
            print("CODE:", res.ground_truth_code)
            for desc in res.correct_descriptions:
                print(desc.nl, " ||| ", desc.parameterization.nl, "|||", desc.parameterization.code)

            print("---")


if __name__ == "__main__":
    fire.Fire({
        "run_autodoc": run_autodoc,
        "analyze": analyze,
    })
