import difflib
import io
import json
import textwrap
import time
import tokenize
from typing import Set, Dict, List, Tuple

import tqdm
import fire

import pandas as pd
from databutler.pat import astlib
from databutler.utils import multiprocess, pickleutils, code as codeutils
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook
from scripts.mining.kaggle.static_analysis.type_inference import run_mypy
from scripts.mining.kaggle.notebooks import utils as nb_utils


def tokenize_code(code: str):
    return [tok.string.strip() for tok in tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
            if tok.string.strip() not in [""]]


def _templatize(entry: Dict) -> str:
    code = entry['code']
    code_ast = astlib.parse(code)

    if any(isinstance(stmt, astlib.Assign) for stmt in astlib.iter_stmts(code_ast)):
        code = astlib.to_code(next(iter(stmt for stmt in astlib.iter_stmts(code_ast)
                                        if isinstance(stmt, astlib.Assign))).value)
        code_ast = astlib.parse(code)

    name_repl: Set[astlib.Name] = set()
    lit_repl: Dict[astlib.BaseExpression, str] = {}

    untouched_lits: Set[str] = {'0', '1', 'True', 'False', ''}
    for node in astlib.iter_true_exprs(code_ast, code_ast):
        if isinstance(node, astlib.Name):
            if node.value not in ['True', 'False']:
                name_repl.add(node)

        elif astlib.is_constant(node) and astlib.to_code(node).strip() not in untouched_lits:
            if isinstance(node, astlib.BaseNumber):
                t = "Number"
            elif isinstance(node, astlib.BaseString):
                t = "String"
            else:
                t = "Literal"

            lit_repl[node] = t

    for node in astlib.walk(code_ast):
        if isinstance(node, astlib.Call) and isinstance(node.func, astlib.Name):
            name_repl.discard(node.func)

    new_code_ast = astlib.with_deep_replacements(code_ast, {
        **{name: astlib.create_name_expr("VAR") for name in name_repl},
        **{lit: astlib.create_name_expr(f"LIT_{lit_type}") for lit, lit_type in lit_repl.items()}
    })

    #  Normalize keyword argument order
    node_repl = {}
    for node in astlib.walk(new_code_ast):
        if isinstance(node, astlib.Call):
            pos_args = [arg for arg in node.args if arg.keyword is None]
            kw_args = sorted([arg for arg in node.args if arg.keyword is not None], key=lambda x: x.keyword.value)

            node_repl[node] = astlib.with_changes(node, args=pos_args + kw_args)

    new_code_ast = astlib.with_deep_replacements(new_code_ast, node_repl)

    return codeutils.normalize_code(astlib.to_code(new_code_ast))


def templatize_all():
    with pickleutils.PickledCollectionReader("./static_mined_exprs.pkl") as reader, \
            pickleutils.PickledCollectionWriter("./static_mined_exprs_templates.pkl") as writer:

        chunksize = 1000
        found = 0
        succ = exceptions = timeouts = other = 0
        seen: Set[str] = set()
        for idx in tqdm.tqdm(range(0, len(reader), chunksize)):
            chunk = [reader[i] for i in range(idx, min(idx + chunksize, len(reader)))]

            for res in multiprocess.run_tasks_in_parallel_iter(_templatize, chunk,
                                                               use_progress_bar=False,
                                                               num_workers=4,
                                                               timeout_per_task=30):
                if res.is_success():
                    if res.result not in seen:
                        seen.add(res.result)
                        writer.append(res.result)
                        found += 1

                    writer.flush()

                if res.is_success():
                    succ += 1
                elif res.is_exception():
                    exceptions += 1
                elif res.is_timeout():
                    timeouts += 1

            print(f"Found {found} so far")
            print(f"Succ: {succ} Exceptions: {exceptions} Timeouts: {timeouts}")


def templatize_conala():
    with open("/Users/rbavishi/Research/DataButler/autodoc_trial1/conala.json") as f:
        conala_clean = json.load(f)

    with open("/Users/rbavishi/Research/DataButler/autodoc_trial1/conala-full.json") as f:
        conala_full = json.load(f)

    conala_all = conala_clean

    with pickleutils.PickledCollectionWriter("./conala_templates.pkl") as writer:
        chunksize = len(conala_all)
        found = 0
        succ = exceptions = timeouts = other = 0
        seen: Set[str] = set()
        ctr = 0
        for idx in tqdm.tqdm(range(0, len(conala_all), chunksize)):
            chunk = [{
                'code': conala_all[i]['snippet']
            } for i in range(idx, min(idx + chunksize, len(conala_all)))]

            for res in multiprocess.run_tasks_in_parallel_iter(_templatize, chunk,
                                                               use_progress_bar=False,
                                                               num_workers=4,
                                                               timeout_per_task=30):
                ctr += 1
                if res.is_success():
                    print(conala_all[ctr - 1]['snippet'])
                    print(res.result)
                    if res.result not in seen:
                        seen.add(res.result)
                        writer.append(res.result)
                        found += 1

                    writer.flush()

                if res.is_success():
                    succ += 1
                elif res.is_exception():
                    exceptions += 1
                elif res.is_timeout():
                    timeouts += 1

            print(f"Found {found} so far")
            print(f"Succ: {succ} Exceptions: {exceptions} Timeouts: {timeouts}")


def templatize_dynamic():
    with open("/Users/rbavishi/Research/DataButler/autodoc_trial1/gen_desc_04_17_2022_20_03_48.yaml") as f:
        dynamic_clean = json.load(f)

    dynamic_clean = dynamic_clean

    with pickleutils.PickledCollectionWriter("./dynamic_templates.pkl") as writer:
        chunksize = len(dynamic_clean)
        found = 0
        succ = exceptions = timeouts = other = 0
        seen: Set[str] = set()
        ctr = 0
        for idx in tqdm.tqdm(range(0, len(dynamic_clean), chunksize)):
            chunk = [{
                'code': dynamic_clean[i]['snippet']
            } for i in range(idx, min(idx + chunksize, len(dynamic_clean)))]

            for res in multiprocess.run_tasks_in_parallel_iter(_templatize, chunk,
                                                               use_progress_bar=False,
                                                               num_workers=4,
                                                               timeout_per_task=30):
                ctr += 1
                if res.is_success():
                    print(dynamic_clean[ctr - 1]['snippet'])
                    print(res.result)
                    if res.result not in seen:
                        seen.add(res.result)
                        writer.append(res.result)
                        found += 1

                    writer.flush()

                if res.is_success():
                    succ += 1
                elif res.is_exception():
                    exceptions += 1
                elif res.is_timeout():
                    timeouts += 1

            print(f"Found {found} so far")
            print(f"Succ: {succ} Exceptions: {exceptions} Timeouts: {timeouts}")

def compare():
    with pickleutils.PickledCollectionReader("./static_mined_exprs_saved_templates.pkl") as reader:
        mined_templates = list(reader)

    with pickleutils.PickledCollectionReader("./conala_templates.pkl") as reader:
        conala_templates = list(reader)

    thresholds = [1.0, 0.99, 0.95, 0.9, 0.8]
    thresholds = [0.9]
    prev = None
    cur = set()
    for threshold in thresholds:
        succ = 0
        for c_code in tqdm.tqdm(conala_templates):
            matches = difflib.get_close_matches(c_code, mined_templates, cutoff=threshold, n=3)
            if len(matches) > 0:
                cur.add(c_code)
                is_exact = any(i == c_code for i in matches)
                print(f"QUERY: {c_code} (EXACT={is_exact})")
                print("---")
                for m in matches:
                    print(m)

                print("===")

                succ += 1

        prev = cur.copy()
        cur.clear()

        print(f"Threshold {threshold} Succ: {succ * 100 / len(conala_templates):.2f}")

    print(len(mined_templates))
    print(len(conala_templates))
    print(len(set(conala_templates) & set(mined_templates)))


def analyze():
    with pickleutils.PickledCollectionReader("./static_mined_exprs.pkl") as reader:
        print(len(reader))
        for i in reader:
            print(codeutils.normalize_code(i['code']), i['kind'])
            for k, v in i['type_map'].items():
                print(f"  - {k}: {', '.join(['|'.join(vv) for vv in v])}")


if __name__ == "__main__":
    # from scripts.mining.kaggle.notebooks.utils import retrieve_notebook_data
    # from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook
    # nb = KaggleNotebook("adityasingh3519", "why-never-use-pandas-get-dummies")
    #
    # mine_pandas_expressions(astlib.to_code(nb.get_astlib_ast()))
    # # mine_pandas_expressions(test_code)
    fire.Fire()
    # print(_templatize({'code': "df['D'] = df['B']"}))
