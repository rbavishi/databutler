import textwrap
import time
from typing import Set, Dict, List, Tuple

import tqdm

import pandas as pd
from databutler.pat import astlib
from databutler.utils import multiprocess, pickleutils
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook
from scripts.mining.kaggle.static_analysis.type_inference import run_mypy
from scripts.mining.kaggle.notebooks import utils as nb_utils


def mine_pandas_expressions(code: str, owner: str, slug: str):
    s = time.time()
    code_ast, inferred_types = run_mypy(code)
    found: Set[astlib.BaseExpression] = set()

    #  First find the expressions that evaluate to a dataframe / series
    df_series_exprs: Set[astlib.BaseExpression] = set()
    for node in astlib.walk(code_ast):
        if not isinstance(node, astlib.BaseExpression):
            continue

        expr: astlib.BaseExpression = node
        req_types: Set[str] = {
            'pandas.core.frame.DataFrame',
            'pandas.core.series.Series',
        }
        if expr in inferred_types and not inferred_types[expr].isdisjoint(req_types):
            df_series_exprs.add(expr)

    found.update(df_series_exprs)

    #  Find any function calls that return something else but correspond to Pandas API (example fillna(inplace=True))
    api_usage_exprs: Set[astlib.BaseExpression] = set()
    for expr in df_series_exprs:
        parent = astlib.get_parent(expr, code_ast)
        if parent not in df_series_exprs and isinstance(parent, astlib.Attribute) and parent.value is expr:
            try:
                attr_name = parent.attr.value
                if any(i is not None
                       for i in [getattr(pd.DataFrame, attr_name, None), getattr(pd.Series, attr_name, None)]):
                    outer = astlib.get_parent(parent, code_ast)
                    if isinstance(outer, astlib.Call):
                        if outer not in api_usage_exprs:
                            api_usage_exprs.add(outer)
                    else:
                        api_usage_exprs.add(parent)
            except:
                pass

    found.update(api_usage_exprs)

    #  Find any function calls that take as argument a series / dataframe
    df_series_arg_call_exprs: Set[astlib.BaseExpression] = set()
    for expr in astlib.walk(code_ast):
        if (not isinstance(expr, astlib.Call)) or expr in found:
            continue

        if isinstance(expr.func, astlib.Name) and expr.func.value in ['print', 'id']:
            continue

        if any(arg.value in df_series_exprs for arg in expr.args):
            df_series_arg_call_exprs.add(expr)

    found.update(df_series_arg_call_exprs)

    def _get_type_map(expr: astlib.BaseExpression):
        type_map: Dict[str, List[Set[str]]] = {}
        for node in astlib.walk(expr):
            if isinstance(node, astlib.BaseExpression) and node in inferred_types:
                rep = astlib.to_code(node).strip()
                if rep not in type_map:
                    type_map[rep] = []

                type_map[rep].append(inferred_types[node])

        return type_map

    result: List[Dict] = []

    # print(f"Found {len({expr for expr in df_series_exprs if not isinstance(expr, astlib.Name)})} df/series expressions")
    for expr in {expr for expr in df_series_exprs if not isinstance(expr, astlib.Name)}:
        # print(astlib.to_code(expr))
        result.append({
            'code': astlib.to_code(expr),
            'kind': 'DF_SERIES_EXPR',
            'type_map': _get_type_map(expr)
        })

    # print(f"\n---\nFound {len(api_usage_exprs)} API usage expressions")
    for expr in {expr for expr in api_usage_exprs if not isinstance(expr, astlib.Name)}:
        # print(astlib.to_code(expr))
        result.append({
            'code': astlib.to_code(expr),
            'kind': 'API_USAGE_EXPR',
            'type_map': _get_type_map(expr)
        })

    # print(f"\n---\nFound {len(df_series_arg_call_exprs)} calls with df/series arguments")
    for expr in {expr for expr in df_series_arg_call_exprs if not isinstance(expr, astlib.Name)}:
        # print(astlib.to_code(expr))
        result.append({
            'code': astlib.to_code(expr),
            'kind': 'CALL_WITH_DF_SERIES_ARGS',
            'type_map': _get_type_map(expr)
        })

    for res in result:
        res["owner"] = owner
        res["slug"] = slug

    return result


def _mp_helper(arg):
    nb: KaggleNotebook = arg
    normalized_code = astlib.to_code(nb.get_astlib_ast())
    return mine_pandas_expressions(normalized_code, nb.owner, nb.slug)


def check():
    with pickleutils.PickledCollectionReader("./static_mined_exprs.pkl") as reader:
        print(len(reader))
        for i in reader:
            print(i['code'], i['kind'])


def run_static_mining():
    with nb_utils.get_local_nb_data_storage_reader() as reader,\
         pickleutils.PickledCollectionWriter("./static_mined_exprs.pkl") as writer:
        all_keys: List[Tuple[str, str]] = list(reader.keys())
        print(f"Found {len(all_keys)} keys")

        chunksize = 10
        found = 0
        succ = exceptions = timeouts = other = 0
        for idx in tqdm.tqdm(range(0, len(all_keys), chunksize)):
            chunk = all_keys[idx: idx + chunksize]
            tasks = [KaggleNotebook.from_raw_data(owner, slug, reader[owner, slug])
                     for owner, slug in chunk]

            for res in multiprocess.run_tasks_in_parallel_iter(_mp_helper, tasks,
                                                               use_progress_bar=True,
                                                               num_workers=2,
                                                               timeout_per_task=30):
                if res.is_success() and isinstance(res.result, list) and len(res.result) > 0:
                    found += len(res.result)
                    for elem in res.result:
                        writer.append(elem)

                    writer.flush()

                if res.is_success():
                    succ += 1
                elif res.is_exception():
                    exceptions += 1
                elif res.is_timeout():
                    timeouts += 1

            print(f"Found {found} so far")
            print(f"Succ: {succ} Exceptions: {exceptions} Timeouts: {timeouts}")


if __name__ == "__main__":
    # from scripts.mining.kaggle.notebooks.utils import retrieve_notebook_data
    # from scripts.mining.kaggle.notebooks.notebook import KaggleNotebook
    # nb = KaggleNotebook("adityasingh3519", "why-never-use-pandas-get-dummies")
    #
    # mine_pandas_expressions(astlib.to_code(nb.get_astlib_ast()))
    # # mine_pandas_expressions(test_code)
    run_static_mining()
    # check()
