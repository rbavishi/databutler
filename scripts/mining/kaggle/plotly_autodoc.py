import ast
import collections
import datetime
import glob
import io
import itertools
import os.path
import random
import textwrap
from typing import Tuple, Dict, List, Set, Union, Optional, Any

import astunparse
import attrs
import click
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs
import tqdm
import yaml

from databutler.datana.generic.autodoc import code2nl, nl2code
from databutler.datana.generic.autodoc.few_shot import FewShotExampleCodeAndNL
from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.pat import astlib
from databutler.utils import pickleutils
from databutler.utils.logging import logger

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PLOTLY_MINER_NAME = 'PlotlyMiner'
PLOTLY_CODE_OUTFILE = 'viz_functions.yaml'
DATANA_FUNC_FILENAME = "datana_funcs.db"
DATANA_DF_FILENAME = "datana_dfs.db"


def _get_file_size_in_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _cmp_values(val1, val2) -> bool:
    if type(val1) is not type(val2):
        return False

    if isinstance(val1, (list, tuple)):
        return len(val1) == len(val2) and all(_cmp_values(e1, e2) for e1, e2 in zip(val1, val2))

    elif isinstance(val1, set):
        if len(val1) != len(val2):
            return False

        for e1 in val1:
            if all(not _cmp_values(e1, e2) for e2 in val2):
                return False

        return True

    elif isinstance(val1, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False

        return all(_cmp_values(val1[k], val2[k]) for k in val1.keys())

    elif isinstance(val1, np.ndarray):
        return np.array_equal(val1, val2)

    elif isinstance(val1, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(val1, val2, check_dtype=False, check_like=True)
            return True
        except AssertionError:
            return False

    elif isinstance(val1, pd.Series):
        try:
            pd.testing.assert_series_equal(val1, val2, check_dtype=False)
            return True
        except AssertionError:
            return False

    elif isinstance(val1, pd.Index):
        try:
            pd.testing.assert_index_equal(val1, val2)
            return True
        except AssertionError:
            return False

    elif isinstance(val1, plt.Figure):
        buf1 = io.BytesIO()
        buf2 = io.BytesIO()
        val1.savefig(buf1, format='png', bbox_inches='tight')
        val2.savefig(buf2, format='png', bbox_inches='tight')

        buf1.seek(0)
        buf2.seek(0)
        print(buf1, buf2)
        return buf1.read() == buf2.read()

    elif isinstance(val1, plotly.graph_objs.Figure):
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        val1.write_json(buf1)
        val2.write_json(buf2)
        buf1.seek(0)
        buf2.seek(0)
        return buf1.read() == buf2.read()

    else:
        try:
            return val1 == val2
        except:
            raise TypeError(f"Cannot compare values of type {type(val1)}")


@attrs.define(eq=False, repr=False)
class CodeMiningResult:
    raw: Dict
    results_path: str

    @property
    def data(self):
        return self.raw

    @property
    def code(self) -> str:
        return self.data['code']

    @property
    def col_args(self) -> Dict[str, Union[str, List[str]]]:
        return self.data['col_args']

    @property
    def args_size_mb(self) -> float:
        return sum(_get_file_size_in_mb(os.path.join(self.results_path, v))
                   for v in self.data['df_args'].values())

    @property
    def lazy_df_args(self) -> Dict[str, str]:
        return self.data['df_args']

    @property
    def df_args(self) -> Dict[str, pd.DataFrame]:
        return {
            k: pickleutils.smart_load(os.path.join(self.results_path, v))
            for k, v in self.data['df_args'].items()
        }

    @property
    def username_slug(self) -> str:
        return self.results_path.split(os.path.sep)[-2]


@attrs.define(eq=False, repr=False)
class AutodocDatanaFunction:
    """
    A wrapper around a datana function with additional convenience methods for running autodoc
    """
    func: DatanaFunction

    @property
    def func_type(self) -> str:
        """
        Returns one of "DF_WRITE", "PRINT_EXPR", and "DF_ASSIGN"
        """
        return self.func.metadata["snippet_type"]

    def get_imports(self) -> str:
        import_code_lines: List[str] = []
        for node in ast.walk(ast.parse(self.func.code_str)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_code_lines.append(astunparse.unparse(node).strip())

        return "\n".join(import_code_lines)

    def get_kw_args_copy(self) -> Dict[str, Any]:
        return {
            k: v.copy() if isinstance(v, pd.DataFrame) else v
            for k, v in self.func.get_kw_args().items()
        }

    def _execute_code_with_func_args(self, code: str, func_name: Optional[str] = None) -> plotly.graph_objs.Figure:
        if func_name is None:
            func_name = self.func.func_name

        g = {}
        exec(code, g, g)
        fn = g.get(func_name, None)
        if fn is None:
            raise ValueError(f"Did not find function with name {func_name}")

        kw_args = self.get_kw_args_copy()
        np.random.seed(42)
        return fn(**kw_args)

    def execute(self) -> Any:
        """
        Gets the result of executing the function on its associated arguments
        """
        result = self._execute_code_with_func_args(self.func.code_str)
        return result

    def is_equivalent(self, code: str, func_result: Optional = None) -> bool:
        """
        Checks if the result of executing the given code on the function's arguments yields the same result.

        If `func_result` is not None, it is assumed to be the result of executing the function. This is useful
        for avoiding repeated computation / caching.
        """
        if func_result is None:
            func_result = self.execute()

        result = self._execute_code_with_func_args(code)

        try:
            return _cmp_values(func_result, result)
        except ValueError:
            return False
        except Exception as e:
            logger.warning("Encountered error in value comparison")
            logger.exception(e)
            return False


def _load_code_results(results_path: str) -> List[CodeMiningResult]:
    with open(os.path.join(results_path, PLOTLY_CODE_OUTFILE), 'r') as f:
        return [
            CodeMiningResult(raw, results_path) for raw in yaml.full_load(f)['viz_functions']
        ]


def _get_code_length(code: str) -> int:
    return sum(1 for node in astlib.walk(astlib.parse(code))
               if isinstance(node, astlib.BaseSmallStatement))


def _get_code_lines(code: str) -> int:
    return len(astunparse.unparse(ast.parse(code)).strip().split("\n"))


def create_datana_functions(campaign_dir: str, path_to_paths: str) -> None:
    """
    Collects raw plotly mining data, and creates a single database of datana functions

    Args:
        campaign_dir: Path to an output directory
        path_to_results: Path to the results directory. The directory structure must be of the form
            <kaggle-username>-<notebook-slug>/PlotlyMiner
    """

    #  --------
    #  Collect all the non-empty results
    #  --------

    results_path_mapping: Dict[Tuple[str, str], str] = {}
    ignored: Set[Tuple[str, str]] = set()
    for path_to_results in open(path_to_paths, 'r'):
        path_to_results = path_to_results.replace("\n", "")
        print(path_to_results)
        for path in glob.glob(os.path.join(path_to_results, PLOTLY_MINER_NAME)):
            username_slug = path.split(os.path.sep)[-2]
            if len(_load_code_results(path)) > 0:
                results_path_mapping[username_slug] = path
            else:
                ignored.add(path)

    logger.info(f"Found {len(results_path_mapping)} non-empty notebook mining results "
                f"out of {len(results_path_mapping) + len(ignored)} total.")

    #  --------
    #  Convert into datana functions and put into a single file
    #  --------

    #  First gather all the results, after some basic filtering
    all_code_results: List[CodeMiningResult] = []
    ignored: List[CodeMiningResult] = []
    for results_path in tqdm.tqdm(results_path_mapping.values(), desc="Processing Results"):
        code_results = _load_code_results(results_path)

        for code_res in code_results:
            if _get_code_length(code_res.code) > 10:
                #  Not more than 10 statements
                #  NOTE: These are not text lines, rather the number of AST statement nodes
                ignored.append(code_res)
                continue

            if len(code_res.col_args) + len(code_res.lazy_df_args) > 5:
                #  Ignore too many args
                ignored.append(code_res)
                continue

            all_code_results.append(code_res)

    logger.info(f"Total: {len(all_code_results) + len(ignored)} "
                f"Retained: {len(all_code_results)} "
                f"Unique Retained: {len(set(res.code for res in all_code_results))} "
                f"Ignored: {len(ignored)}")

    #  Convert into datana functions and store to a single pickle collection (+ another for dfs)
    datana_func_path = os.path.join(campaign_dir, DATANA_FUNC_FILENAME)
    datana_df_path = os.path.join(campaign_dir, DATANA_DF_FILENAME)

    username_slug_set: Dict[str, int] = collections.defaultdict(int)
    verif_map: Dict[CodeMiningResult, int] = {}

    os.makedirs(campaign_dir, exist_ok=True)
    num_written = 0

    with pickleutils.PickledCollectionWriter(datana_func_path, overwrite_existing=True) as code_writer, \
            pickleutils.PickledCollectionWriter(datana_df_path, overwrite_existing=True) as df_writer:
        df_lazy_refs: Dict[str, pickleutils.PickledRef] = {}

        for res in tqdm.tqdm(all_code_results, desc='Datana Functions Conv.'):
            res: CodeMiningResult
            lazy_df_args = res.lazy_df_args
            df_args = res.df_args

            kw_args: Dict[str, Union[str, pickleutils.PickledRef]] = {}
            #  Add the dataframe arguments.
            for arg in lazy_df_args.keys():
                key = f"{res.results_path}/{lazy_df_args[arg]}"
                if key not in df_lazy_refs:
                    df_val = df_args[arg]
                    df_lazy_refs[key] = pickleutils.PickledRef(datana_df_path, index=len(df_writer))
                    df_writer.append(df_val)

                ref = df_lazy_refs[key]
                kw_args[arg] = ref

            #  Update with the column arguments.
            kw_args.update(res.col_args)

            #  We now have all the ingredients for a datana function
            ctr = username_slug_set[res.username_slug]
            username_slug_set[res.username_slug] += 1

            func = DatanaFunction(
                code_str=res.code,
                uid=f"{res.username_slug}_{ctr}",
                func_name="viz",
                pos_args=[],
                kw_args=kw_args,
                metadata={
                    **res.data,
                    "username_slug": res.username_slug,
                }
            )

            #  OPTIONAL: Retain the mapping to enable integrity checks
            verif_map[res] = len(code_writer)

            #  Write to the db
            code_writer.append(func)
            num_written += 1

    #  OPTIONAL: Run correctness / integrity checks.
    with pickleutils.PickledCollectionReader(datana_func_path) as code_reader:
        for code_res, idx in tqdm.tqdm(verif_map.items(), desc='Verifying'):
            df_args = code_res.df_args
            func = code_reader[idx]
            func_kw_args = func.get_kw_args()

            for arg, df in df_args.items():
                func_df = func_kw_args[arg]
                assert func_df.shape == df.shape
                assert list(func_df.columns) == list(df.columns)

    logger.info(f"Wrote {num_written} datana functions and dfs to "
                f"{datana_func_path} and {datana_df_path} respectively")


def _load_datana_functions(campaign_dir: str) -> List[DatanaFunction]:
    all_functions: List[DatanaFunction] = []

    datana_func_path = os.path.join(campaign_dir, DATANA_FUNC_FILENAME)

    with pickleutils.PickledCollectionReader(datana_func_path) as func_reader:
        all_functions.extend(func_reader[i] for i in range(0, len(func_reader)))

    return all_functions


def _uniqify_functions(all_functions: List[DatanaFunction]) -> List[List[DatanaFunction]]:
    """
    Computes equivalence classes of functions, based on some evaluation criteria.
    """
    # TODO: For now this is purely syntactic. We can incorporate equivalence modulo variable renaming,
    #       or even semantic equivalence.
    per_code_mapping: Dict[str, List[DatanaFunction]] = collections.defaultdict(list)
    for func in all_functions:
        per_code_mapping[astunparse.unparse(ast.parse(func.code_str)).strip()].append(func)

    return list(per_code_mapping.values())


def prepare_few_shot(campaign_dir: str, num_examples: int = 5, version: int = 1, strategy: str = "random"):
    """
    Given a campaign where datana functions have been generated, prepare a list of few-shot examples as a YAML file.
    You should fill out the NL fields for each of the example before you use this for autodoc

    Args:
        campaign_dir: Path to an output directory (same as the one used for create_datana_functions)
        num_examples: Number of examples to generate
        version: An integer corresponding to the version number.
        strategy: One of ['random', ]

    Returns:
        None. Generates the file {campaign_dir}/few_shot_{version}.yaml
    """
    few_shot_path = os.path.join(campaign_dir, f"few_shot_{version}.yaml")

    if os.path.exists(few_shot_path) and not click.confirm(f"Do you want to overwrite version {version}?"):
        return

    logger.info("Loading datana functions")
    all_functions: List[DatanaFunction] = _load_datana_functions(campaign_dir)
    logger.info(f"Found {len(all_functions)} datana functions")

    uniqified_funcs = _uniqify_functions(all_functions)
    logger.info(f"Found {len(uniqified_funcs)} equivalence classes")

    selected: List[DatanaFunction] = []

    if strategy == "random":
        selected.extend(random.sample(all_functions, num_examples))
    else:
        raise NotImplementedError(f"Unknown strategy {strategy}")

    few_shot_examples: List[Dict] = []
    for func in selected:
        code: str = func.code_str
        nl: List[str] = ["TODO"]

        args = func.get_kw_args()
        col_args = {k: v for k, v in args.items() if not isinstance(v, pd.DataFrame)}
        df_args = {k: v for k, v in args.items() if isinstance(v, pd.DataFrame)}

        few_shot_examples.append({
            "code": code,
            "nl": nl,
            "df_args": {
                k: str(v.head(5)) for k, v in sorted(df_args.items(), key=lambda x: x[0])
            },
            "col_args": col_args,
        })

    with open(few_shot_path, "w") as f:
        yaml.dump(few_shot_examples, f)

    logger.info(f"Dumped {len(few_shot_examples)} examples to {few_shot_path}.")


def run_autodoc(campaign_dir: str, num_few_shot: int = 5, few_shot_version: int = 1, temperature: float = 0.0,
                num_tries: int = 3, max_desc_len: int = 3, engine: str = 'code-davinci-001',
                rerun_timestamp: Optional[str] = None):
    """
    Run auto-documentation on all datana functions. Make sure you have written NL for the few-shot examples

    Args:
        campaign_dir: Path to the campaign directory (same as the one used for create_datana_functions and
            prepare_few_shot
        num_few_shot: Number of few-shot examples to used. Capped at the number of few-shot examples provided.
        few_shot_version: Version number of the few-shot examples dump to use.
        temperature: Temperature for the model
        num_tries: Number of times to attempt generation a bidirectionally consistent description. Only makes sense
            if temperature > 0
        max_desc_len: Maximum number of bullets to generate as part of a description
        engine: The Codex engine to use. Do not change from default unless you know what you're doing.
        rerun_timestamp: Timestamp of the run to resume/overwrite. Useful to recover from errors.

    Returns:
        None. Generates a file "gen_desc_{timestamp}.yaml" containing descriptions for datana functions for which
            the auto-doc process was successful.

    """
    logger.info("Loading datana functions")
    all_functions: List[DatanaFunction] = _load_datana_functions(campaign_dir)
    logger.info(f"Found {len(all_functions)} datana functions")

    uid_to_func: Dict[str, DatanaFunction] = {func.uid: func for func in all_functions}

    few_shot_path = os.path.join(campaign_dir, f"few_shot_{few_shot_version}.yaml")
    if not os.path.exists(few_shot_path):
        raise FileNotFoundError(f"Few-shot examples not found at {few_shot_path}")

    with open(few_shot_path, "r") as f:
        few_shot_examples = yaml.full_load(f)

    if any("TODO" in ex["nl"] for ex in few_shot_examples):
        raise ValueError(f"Few-shot examples missing NL description")

    uniqified_funcs = _uniqify_functions(all_functions)
    logger.info(f"Found {len(uniqified_funcs)} equivalence classes")

    #  If temperature is 0, we are going to get the same result everytime. So no point in retrying.
    num_tries = 1 if temperature == 0.0 else num_tries
    succ = fail = 0
    generated_descriptions: Dict[DatanaFunction, List[str]] = {}
    if rerun_timestamp is not None:
        cur_timestamp = rerun_timestamp
    else:
        cur_timestamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

    generated_descs_output_path = os.path.join(campaign_dir, f"gen_desc_{cur_timestamp}.yaml")
    generation_logs_dir = os.path.join(campaign_dir, f"autodoc_{cur_timestamp}")

    already_processed: Set[str] = set()
    if os.path.exists(generated_descs_output_path) and click.confirm(f"Only run previously failed/skipped?"):
        with open(generated_descs_output_path, "r") as f:
            for res in yaml.full_load(f):
                generated_descriptions[uid_to_func[res['uid']]] = [i[2:] for i in res['desc'].split("\n")]
                already_processed.add(res['uid'])

    def _dump_generated_descriptions():
        with open(generated_descs_output_path, 'w') as f:
            yaml.dump([{
                "uid": fn.uid,
                "code": fn.code_str,
                "desc": "\n".join(f"* {b}" for b in bullets)
            } for fn, bullets in generated_descriptions.items()], f)

    os.makedirs(generation_logs_dir, exist_ok=True)

    cur_log_handle = None

    #  Choose a representative for each class.
    #  All functions in the same equivalence class will share a description.
    func_rep: Dict[DatanaFunction, DatanaFunction] = {}
    for eq_class in uniqified_funcs:
        representative = eq_class[0]
        for func in eq_class:
            if func.uid in already_processed:
                representative = func
                break

        for func in eq_class:
            func_rep[func] = representative

    to_process = [func for func in set(func_rep.values()) if func.uid not in already_processed]
    to_process = sorted(to_process, key=lambda x: len(x.code_str))

    logger.info(f"Generating descriptions for {len(to_process)} representative functions")

    with tqdm.tqdm(to_process, desc='Generating NL', dynamic_ncols=True) as pbar:
        for func in pbar:
            if cur_log_handle is not None:
                logger.remove(cur_log_handle)

            log_path = os.path.join(generation_logs_dir, f"{func.uid}.log")
            cur_log_handle = logger.add(log_path, level="TRACE")

            logger.info(f"Code for {func.uid}: \n{func.code_str}")
            wrapper = AutodocDatanaFunction(func)
            try:
                expected_result = wrapper.execute()

            except Exception as e:
                logger.warning(f"Failed to run datana function {func.uid}")
                logger.exception(e)
                fail += 1
                pbar.set_postfix(succ=succ, fail=fail)
                continue
            else:
                if expected_result is None:
                    logger.warning(f"Datana function {func.uid} returned None")
                    fail += 1
                    pbar.set_postfix(succ=succ, fail=fail)
                    continue

            few_shot_c2nl: List[FewShotExampleCodeAndNL] = [
                FewShotExampleCodeAndNL(code=ex["code"], nl=ex["nl"])
                for ex in few_shot_examples
            ]

            few_shot_nl2c: List[FewShotExampleCodeAndNL] = [
                FewShotExampleCodeAndNL(code=ex["code"],
                                        nl=ex["nl"])
                for ex in few_shot_examples
            ]

            func_sig_str = func.code_str.strip().split('\n')[0]
            c2nl_engine = code2nl.SimpleCodeToNatLang(temperature=temperature, engine=engine, max_tokens=256)
            nl2c_engine = nl2code.SimpleNatLangToCode(temperature=0.0, engine=engine, max_tokens=512)

            found = False
            for attempt in range(1, num_tries + 1):
                #  Generate one NL description bullet at a time, and check for consistency.
                c2nl_task = code2nl.CodeToNatLangTask(
                    few_shot_examples=few_shot_c2nl[:num_few_shot],
                    target_code=func.code_str,
                    task_description="Describe the following data science code snippets in plain english as a "
                                     "sequence of bullets"
                )

                collected_bullets: List[str] = []
                for bullet in itertools.islice(c2nl_engine.get_nl_bullets(c2nl_task), max_desc_len):
                    logger.trace(f"\nAttempt {attempt} Bullet {len(collected_bullets) + 1}: {bullet}")
                    collected_bullets.append(bullet)

                    #  Check if this is enough to reproduce the code (semantic equivalence check)
                    nl2c_task = nl2code.NatLangToCodeTask(
                        few_shot_examples=few_shot_nl2c,
                        target_nl=collected_bullets,
                        task_description="Generate a data science code snippet given the description",
                        output_prefix=func_sig_str,
                    )

                    new_code = nl2c_engine.get_code(nl2c_task)
                    new_code = f"{func_sig_str}\n" \
                               f"{textwrap.indent(wrapper.get_imports(), ' ' * 4)}\n" \
                               f"{new_code[len(func_sig_str) + 1:]}"
                    logger.trace(f"Generated Code:\n {new_code}")

                    try:
                        if wrapper.is_equivalent(new_code, func_result=expected_result):
                            generated_descriptions[func] = collected_bullets
                            full_desc = "\n".join(f"* {b}" for b in collected_bullets)
                            logger.info(f"Found Description for {func.uid}:\n{full_desc}")
                            found = True
                            break

                    except Exception as e:
                        pass

                if found:
                    succ += 1
                    _dump_generated_descriptions()
                    break
            else:
                logger.info(f"Could not generate description for {func.uid}")
                fail += 1

            pbar.set_postfix(succ=succ, fail=fail)

        _dump_generated_descriptions()
        logger.info(f"All descriptions saved at {generated_descs_output_path}")


def analyze_autodoc_results(campaign_dir: str, timestamp: str):
    all_functions: List[DatanaFunction] = _load_datana_functions(campaign_dir)
    uniqified_funcs = _uniqify_functions(all_functions)

    uid_to_func: Dict[str, DatanaFunction] = {func.uid: func for func in all_functions}

    results_file = os.path.join(campaign_dir, f"gen_desc_{timestamp}.yaml")
    with open(results_file, 'r') as f:
        results = yaml.full_load(f)

    found: Set[str] = {res['uid'] for res in results}
    not_found: Set[str] = set()
    for eq_class in uniqified_funcs:
        if any(func.uid in found for func in eq_class):
            continue

        not_found.add(eq_class[0].uid)

    with open(os.path.join(campaign_dir, f"report_{timestamp}.txt"), "w") as rp:
        print(f"Success: {len(found)} Failed: {len(not_found)}", file=rp)
        code_sizes_succ = [_get_code_length(uid_to_func[i].code_str) for i in found]
        code_sizes_fail = [_get_code_length(uid_to_func[i].code_str) for i in not_found]

        code_lines_succ = [_get_code_lines(uid_to_func[i].code_str) for i in found]
        code_lines_fail = [_get_code_lines(uid_to_func[i].code_str) for i in not_found]

        print(f"Code Sizes (Success): {pd.Series(code_sizes_succ).describe()}", file=rp)
        print(f"Code Sizes (Failure): {pd.Series(code_sizes_fail).describe()}", file=rp)

        print(f"Code Lines (Success): {pd.Series(code_lines_succ).describe()}", file=rp)
        print(f"Code Lines (Failure): {pd.Series(code_lines_fail).describe()}", file=rp)

        print(f"=======\nFailures\n=======", file=rp)
        for uid in sorted(not_found, key=lambda x: len(uid_to_func[x].code_str)):
            print(f"{uid}:\n{uid_to_func[uid].code_str}", file=rp)
            print("-------", file=rp)

            log_path = os.path.join(campaign_dir, f"autodoc_{timestamp}", f"{uid}.log")
            if os.path.exists(log_path):
                with open(log_path) as log_fp:
                    print(log_fp.read(), file=rp)
            else:
                print(f"No logs found for {uid}", file=rp)
            print("-------", file=rp)


if __name__ == "__main__":
    def str_presenter(dumper, data):
        if len(data.splitlines()) > 1:  # check for multiline string
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)


    yaml.add_representer(str, str_presenter)

    fire.Fire({
        'create_datana_functions': create_datana_functions,
        'prepare_few_shot': prepare_few_shot,
        'run_autodoc': run_autodoc,
        'analyze_autodoc_results': analyze_autodoc_results,
    })
