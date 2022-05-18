import collections
import itertools
import os
from typing import Optional, List, Dict, Any, Tuple, Set

import attrs
import fire
import pandas as pd
import tqdm
import yaml

from databutler.datana.generic.autodoc import code2nl, nl2code
from databutler.datana.generic.autodoc.few_shot import FewShotExampleCodeAndNL
from databutler.mining.kaggle.static_analysis.pandas_autodoc_utils import find_instantiation_map
from databutler.mining.kaggle.static_analysis.pandas_mining import MINING_RESULTS_FILE
from databutler.mining.kaggle.static_analysis.pandas_mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import pickleutils, langmodels, code as codeutils

ENGINE = 'code-davinci-002'
PREPROCESSING_RESULTS_FILE = "pandas_mining_preprocessed.pkl"


@attrs.define(eq=False, repr=False)
class AutodocFewShotExample:
    code: str
    nl: str

    @classmethod
    def from_json(cls, val_json: Dict) -> 'AutodocFewShotExample':
        return AutodocFewShotExample(code=val_json['code'], nl=val_json['nl'])

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "nl": self.nl,
        }


@attrs.define(eq=False, repr=False)
class AutodocDescription:
    success: bool
    nl: str
    generated_code: str
    code_parseable: bool
    assistance_level: int

    parameterized_nl: Optional[str] = attrs.field(default=None)
    parameterized_code: Optional[str] = attrs.field(default=None)


@attrs.define(eq=False, repr=False)
class AutodocResult:
    uid: str
    success: bool
    ground_truth_code: str

    correct_descriptions: List[AutodocDescription]
    incorrect_descriptions: List[AutodocDescription]


def normalize_code_for_comparison(code: str, df_args: Set[str]):
    code_ast = astlib.parse(code)
    #  Normalize keyword argument order
    node_repl = {}
    for node in astlib.walk(code_ast):
        if isinstance(node, astlib.Call):
            pos_args = [arg for arg in node.args if arg.keyword is None]
            kw_args = sorted([
                astlib.with_changes(arg, comma=astlib.cst.MaybeSentinel.DEFAULT)
                for arg in node.args if arg.keyword is not None
            ], key=lambda x: x.keyword.value)
            if len(kw_args) > 0:
                node_repl[node] = astlib.with_changes(node, args=pos_args + kw_args)

    if len(node_repl) > 0:
        code_ast = astlib.with_deep_replacements(code_ast, node_repl)

    #  Replace attribute-based column access to the best of our ability
    attr_repl = {}
    for node in astlib.walk(code_ast):
        if isinstance(node, astlib.Attribute) and isinstance(node.value, astlib.Name) and node.value.value in df_args:
            if not hasattr(pd.DataFrame, node.attr.value):
                new_node = astlib.parse_expr(f"{node.value.value}[\"{node.attr.value}\"]")
                attr_repl[node] = new_node

    if len(attr_repl) > 0:
        code_ast = astlib.with_deep_replacements(code_ast, attr_repl)

    return codeutils.normalize_code_fast(astlib.to_code(code_ast))


def get_few_shot_example_path(campaign_dir: str, version: int) -> str:
    return os.path.join(campaign_dir, f"few_shot_{version}.yaml")


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


def validate_nl_descriptions2(
        batch: List[MinedResult],
        few_shot_examples: List[AutodocFewShotExample],
        batch_candidates: List[List[str]],
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
) -> List[AutodocResult]:
    few_shot_nl2c: List[FewShotExampleCodeAndNL] = [
        FewShotExampleCodeAndNL(code=ex.code, nl=ex.nl)
        for ex in few_shot_examples
    ]

    df_arg_names_list: List[Set[str]] = [set(snippet.df_vars) for snippet in batch]
    gts = [normalize_code_for_comparison(snippet.code, df_arg_names)
           for snippet, df_arg_names in zip(batch, df_arg_names_list)]

    gt_tokens_list: List[List[int]] = [langmodels.codex_tokenize(snippet.code)["token_ids"]
                                       for snippet in batch]

    nl2c_engine = nl2code.SimpleNatLangToCode(temperature=0.0, engine=ENGINE,
                                              max_tokens=max(len(v) for v in gt_tokens_list) + 64)

    #  First try without logit biasing
    nl2c_tasks = [
        nl2code.NatLangToCodeTask(
            few_shot_examples=few_shot_nl2c,
            target_nl=candidate,
            task_description="Generate a Python pandas code snippet given the english description",
        ) for candidates in batch_candidates for candidate in candidates
    ]

    code_results_batch: List[str] = nl2c_engine.parallel_get_code(nl2c_tasks, key_manager=key_manager)
    start_idx: int = 0
    autodoc_results: List[AutodocResult] = []

    for snippet, candidates, gt, gt_tokens, df_arg_names in zip(
            batch, batch_candidates, gts, gt_tokens_list, df_arg_names_list
    ):
        correct: List[AutodocDescription] = []
        incorrect: List[AutodocDescription] = []

        code_results: List[str] = code_results_batch[start_idx: start_idx + len(candidates)]
        for assistance_level in [0, 1]:
            for nl, code_result in zip(candidates, code_results):
                try:
                    code_result = normalize_code_for_comparison(code_result, df_arg_names)
                except (astlib.cst.ParserSyntaxError, SyntaxError):
                    is_equiv = False
                    parseable = False
                else:
                    parseable = True
                    is_equiv = gt == code_result

                desc = AutodocDescription(
                    success=is_equiv,
                    nl=nl,
                    generated_code=code_result,
                    assistance_level=assistance_level,
                    code_parseable=parseable,
                )
                (correct if is_equiv else incorrect).append(desc)

            if len(correct) > 0:
                break
            else:
                #  If it failed, we will try logit-biasing
                nl2c_engine = nl2code.SimpleNatLangToCode(temperature=0.0, engine=ENGINE,
                                                          max_tokens=len(gt_tokens) + 64)
                code_results: List[str] = nl2c_engine.parallel_get_code(
                    nl2c_tasks,
                    allowed_tokens=gt_tokens,
                    key_manager=key_manager,
                )

        print(f"Code: {gt}")
        print("Correct:")
        for desc in correct:
            print(f"NL: {desc.nl} || Generated Code: {desc.generated_code}")
        print("---")
        print("Incorrect:")
        for desc in incorrect:
            print(f"NL: {desc.nl} || Generated Code: {desc.generated_code}")
        print("---")

        autodoc_results.append(AutodocResult(
            uid=snippet.uid,
            success=len(correct) > 0,
            ground_truth_code=gt,
            correct_descriptions=correct,
            incorrect_descriptions=incorrect,
        ))

    return autodoc_results


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
        for nl, code_result in zip(candidates, code_results):
            try:
                code_result = normalize_code_for_comparison(code_result, df_arg_names)
            except (astlib.cst.ParserSyntaxError, SyntaxError):
                is_equiv = False
                parseable = False
            else:
                parseable = True
                is_equiv = gt == code_result

            desc = AutodocDescription(
                success=is_equiv,
                nl=nl,
                generated_code=code_result,
                assistance_level=assistance_level,
                code_parseable=parseable,
            )
            (correct if is_equiv else incorrect).append(desc)

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

    c2nl_key_manager = langmodels.OpenAIKeyManager(keys=available_keys[:1])
    nl2c_key_manager = langmodels.OpenAIKeyManager(keys=available_keys[1:])

    #  Get NL for each in one shot using parallel prompts
    nl_descriptions = get_nl_descriptions_for_batch(
        batch, few_shot_examples, temperature, num_nl_per_query, key_manager=c2nl_key_manager
    )
    num_success = 0
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

    all_lengths = []
    for j in nl_descriptions:
        for desc_candidates in j:
            all_lengths.append(len(langmodels.codex_tokenize(desc_candidates)["token_ids"]))

    print(min(all_lengths), max(all_lengths), sum(all_lengths) / len(all_lengths))
    print(num_success)
    return []


def run_autodoc(
        campaign_dir: str,
        few_shot_version: int = 1,
        batch_size: int = 20,
        num_results: Optional[int] = None,
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

    num_col_accesses = len(templates_to_snippet_dict["DF1[STR1]"])
    print(f"Found {num_col_accesses} col-access representations")

    import random
    random.shuffle(uids_to_process)
    with pickleutils.PickledMapReader(mining_results_path) as reader:
        # for uid in uids_to_process:
        #     t_ast = astlib.parse(reader[uid].template)
        #     c_ast = astlib.parse(reader[uid].code)
        #     print(f"Code: {reader[uid].code}")
        #     print(f"Template: {reader[uid].template}")
        #     find_instantiation_map(t_ast, c_ast)
        #
        # return
        chunk = [reader[key] for key in uids_to_process][:batch_size]
        # chunk = [reader[key] for key in uids_to_process if "read_csv" in reader[key].code and "=" in reader[key].code][:batch_size]
        run_autodoc_for_batch(chunk, few_shot_examples, temperature=0.8, num_nl_per_query=10)


if __name__ == "__main__":
    fire.Fire({
        "run_autodoc": run_autodoc,
    })
