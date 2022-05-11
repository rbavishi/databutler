import contextlib
import io
import itertools
from typing import Dict, Any, List, Set, Optional

import attrs
import fire

from databutler.mining.kaggle.notebooks.notebook import KaggleNotebook
from databutler.mining.kaggle.notebooks import utils as nb_utils
from databutler.mining.kaggle.static_analysis.pandas_mining_utils import (
    find_library_usages,
    find_constants,
    replace_constants,
    DF_TYPE, SERIES_TYPE, GROUPBY_TYPES,
    has_undefined_references,
    normalize_df_series_vars,
    normalize_call_args,
    normalize_col_accesses,
    templatize,
)
from databutler.pat import astlib
from databutler.pat.analysis.type_analysis.inference import run_mypy
from databutler.pat.analysis.type_analysis.mypy_types import SerializedMypyType
from databutler.utils import multiprocess, pickleutils, code as codeutils

JsonDict = Dict


@attrs.define(eq=False, repr=False)
class MinedResult:
    code: str
    template: str
    kind: str
    nb_owner: str
    nb_slug: str
    uid: str

    expr_type: Optional[SerializedMypyType]
    type_map: Dict[str, SerializedMypyType]
    df_vars: List[str]
    series_vars: List[str]
    template_vars: Dict[str, List[str]]

    def to_json(self) -> JsonDict:
        pass

    @classmethod
    def from_json(cls, json_dict: JsonDict) -> 'MinedResult':
        pass

    def prettify(self) -> str:
        with contextlib.redirect_stdout(io.StringIO()) as f_out:
            url = f"https://kaggle.com/{self.nb_owner}/{self.nb_slug}"
            print(f"UID: {self.uid}\nKind: {self.kind}\nURL: {url}")
            print("----------")
            print(f"Code:\n{self.code}")
            print("----------")
            print(f"Templatized:\n{self.template}")
            print("----------")
            print(f"Value Type: {'Any' if self.expr_type is None else self.expr_type.to_string()}")
            print("==========")

        return f_out.getvalue()

    def __repr__(self):
        return self.prettify()

    def __str__(self):
        return self.prettify()


def prepare_mined_result(
        target: astlib.AstNode,
        code_ast: astlib.AstNode,
        inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
        lib_usages: Dict[astlib.Name, str],
        constants: Dict[astlib.BaseExpression, Any],
        kind: str,
        nb_owner: str,
        nb_slug: str,
) -> Optional[MinedResult]:
    # import time
    # s = time.time()
    expr_type = inferred_types.get(target, None)
    true_exprs: Set[astlib.BaseExpression] = set(astlib.iter_true_exprs(target, code_ast))
    target_nodes = set(astlib.walk(target))
    free_vars: Set[astlib.Name] = {a.node for a in astlib.get_definitions_and_accesses(code_ast)[1]
                                   if all(d.enclosing_node not in target_nodes for d in a.definitions)} & true_exprs

    def _fixup_metadata(node_mapping, _target):
        nonlocal inferred_types, lib_usages, true_exprs, free_vars
        inferred_types = {node_mapping.get(k, k): v for k, v in inferred_types.items()}
        lib_usages = {node_mapping.get(k, k): v for k, v in lib_usages.items()}
        _target_exprs = {n: idx for idx, n in enumerate(astlib.walk(_target)) if isinstance(n, astlib.BaseExpression)}
        true_exprs = sorted({node_mapping.get(n, n) for n in true_exprs}.intersection(_target_exprs.keys()),
                            key=lambda x: _target_exprs.get(x, 0))
        free_vars = sorted({node_mapping.get(n, n) for n in free_vars
                            if isinstance(node_mapping.get(n, n), astlib.Name)}.intersection(true_exprs),
                           key=lambda x: _target_exprs.get(x, 0))

    #  Replace any constant expressions
    target, repl_map = replace_constants(target, true_exprs, free_vars, constants)
    _fixup_metadata(repl_map, target)
    # print("CONST REPL", astlib.to_code(target))

    #  Ignore if any undefined non-import, non-df, non-series variables
    if has_undefined_references(target, free_vars, inferred_types, lib_usages):
        # print("SKIPPING", astlib.to_code(target))
        return None

    target, repl_map = normalize_call_args(target, inferred_types)
    _fixup_metadata(repl_map, target)

    #  Normalize df and series variables (these will become parameters)
    target, df_vars, series_vars, repl_map = normalize_df_series_vars(target, true_exprs, free_vars, inferred_types)
    _fixup_metadata(repl_map, target)

    #  Convert attribute-based column accesses to subscript-based accesses.
    target, repl_map = normalize_col_accesses(target, true_exprs, inferred_types)
    _fixup_metadata(repl_map, target)

    #  Create templates for clustering
    template, template_vars_map = templatize(target, true_exprs, free_vars, inferred_types, lib_usages)
    # print("TEMPLATIZED", astlib.to_code(target))

    res = MinedResult(
        code=codeutils.normalize_code_fast(astlib.to_code(target)),
        template=codeutils.normalize_code_fast(astlib.to_code(template)),
        kind=kind,
        nb_owner=nb_owner,
        nb_slug=nb_slug,
        uid="",  # Will be set later
        expr_type=expr_type,
        type_map={codeutils.normalize_code_fast(astlib.to_code(k)): v for k, v in inferred_types.items()},
        df_vars=df_vars,
        series_vars=series_vars,
        template_vars=template_vars_map,
    )
    # print("-----", id(code_ast), time.time() - s)
    return res


def mine_code(code: str, nb_owner: str = "owner", nb_slug: str = "slug") -> List[MinedResult]:
    result: List[MinedResult] = []
    code_ast = astlib.parse(code)
    _, inferred_types = run_mypy(code_ast)

    lib_usages = find_library_usages(code_ast)
    constants = find_constants(code_ast)

    #  1. Find non-name expressions that result in a pandas dataframe, series, or groupby
    df_exprs: Set[astlib.BaseExpression] = set()
    series_exprs: Set[astlib.BaseExpression] = set()
    groupby_exprs: Set[astlib.BaseExpression] = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if node in inferred_types:
            if inferred_types[node].equals(DF_TYPE):
                df_exprs.add(node)
                # print("DF", astlib.to_code(node))
            elif inferred_types[node].equals(SERIES_TYPE):
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

        #  We are looking for a function call whose caller involves a dataframe/series/groupby
        if any(n in df_series_gpby_exprs for n in astlib.iter_true_exprs(node.func, code_ast)):
            api_usage_exprs.add(node)

        #  Also watch out for things like pd.concat
        if any(isinstance(n, astlib.Name) and n in lib_usages and 'pandas' in lib_usages[n]
               for n in astlib.iter_true_exprs(node.func, code_ast)):
            api_usage_exprs.add(node)

    all_found_exprs = df_series_gpby_exprs | api_usage_exprs

    #  3. Find function calls that take dataframe / series / groupby arguments that were not identified previously
    call_exprs_with_df_series_gpby_args = set()
    for node in astlib.iter_true_exprs(code_ast, context=code_ast):
        if isinstance(node, astlib.Call) and node not in all_found_exprs:
            #  We do not want print statements
            if isinstance(node.func, astlib.Name) and node.func.value == 'print':
                continue

            if any(arg.value in df_series_gpby_exprs for arg in node.args):
                call_exprs_with_df_series_gpby_args.add(node)

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
            result.append(prepare_mined_result(expr, code_ast, inferred_types, lib_usages, constants, "DF_EXPR",
                                               nb_owner, nb_slug))

    for expr in series_exprs:
        if not isinstance(expr, astlib.Name):
            result.append(prepare_mined_result(expr, code_ast, inferred_types, lib_usages, constants, "SERIES_EXPR",
                                               nb_owner, nb_slug))

    for expr in api_usage_exprs:
        if not isinstance(expr, astlib.Name):
            result.append(prepare_mined_result(expr, code_ast, inferred_types, lib_usages, constants, "API_USAGE",
                                               nb_owner, nb_slug))

    for expr in call_exprs_with_df_series_gpby_args:
        result.append(prepare_mined_result(expr, code_ast, inferred_types, lib_usages, constants, "CALLED_W_PD_ARGS",
                                           nb_owner, nb_slug))

    result = [res for res in result if res is not None]

    #  Remove multiple entries of DF1[STR1] (column-access) as they are usually too many in number
    col_acc_template = "DF1[STR1]"
    if any(res.template == col_acc_template for res in result):
        col_access_representative = next(res for res in result if res.template == col_acc_template)
        result = [res for res in result if (not res.template == col_acc_template) or res is col_access_representative]

    #  Assign UIDs
    for idx, res in enumerate(result, 1):
        res.uid = f"{nb_owner}/{nb_slug}:{idx}"

    # for res in result:
    #     print(res.code)
    #     print(res.template)
    #     print("-----------")

    # unique_templates = {res.template for res in result}

    # print(f"Found {len(result)} results")
    # print(f"Found {len(unique_templates)} unique templates")
    return result


def mine_notebook(owner: str, slug: str) -> List[MinedResult]:
    nb = KaggleNotebook.from_raw_data(owner, slug, nb_utils.retrieve_notebook_data(owner, slug))
    #  Convert notebook to script
    normalized_code = codeutils.normalize_code_fast(astlib.to_code(nb.get_astlib_ast()))
    mined_results = mine_code(normalized_code, nb.owner, nb.slug)
    return mined_results


def _mine_notebook_mp_helper(nb: KaggleNotebook):
    pass


def start_mining_campaign(campaign_dir: str, num_processes: int = 2):
    pass


if __name__ == "__main__":
    fire.Fire({
        'mine_notebook': mine_notebook
    })
