import builtins
import collections
import contextlib
import glob
import io
import os
import string
from typing import Dict, Any, Tuple, List, Collection, Optional, Set

import attrs

import pandas as pd
from databutler.pat import astlib
from databutler.pat.analysis.type_analysis.mypy_types import SerializedMypyType

DF_TYPE = "pandas.core.frame.DataFrame"
SERIES_TYPE = "pandas.core.series.Series"
DF_GROUPBY_TYPE = "pandas.core.groupby.generic.DataFrameGroupBy"
SERIES_GROUPBY_TYPE = "pandas.core.groupby.generic.SeriesGroupBy"
BASE_GROUPBY_TYPE = "pandas.core.groupby.groupby.GroupBy"
GROUPBY_TYPES = {
    BASE_GROUPBY_TYPE,
    DF_GROUPBY_TYPE,
    SERIES_GROUPBY_TYPE,
}

NewTarget = astlib.AstNode
DfArgs = List[str]
SeriesArgs = List[str]
NodeReplMap = Dict[astlib.AstNode, astlib.AstNode]
JsonDict = Dict

_BUILTIN_FUNCS = {k for k in builtins.__dict__ if not k.startswith("_")}


@attrs.define(eq=False, repr=False)
class MinedResult:
    code: str
    template: str
    kind: str
    uid: str
    reference: str

    expr_type: Optional[SerializedMypyType]
    type_map: Dict[str, SerializedMypyType]
    df_vars: List[str]
    series_vars: List[str]
    template_vars: Dict[str, List[str]]

    extra_context_vars: Dict[str, str] = attrs.field(factory=dict)
    lib_usages: Dict[str, str] = attrs.field(factory=dict)

    def to_json(self) -> JsonDict:
        pass

    @classmethod
    def from_json(cls, json_dict: JsonDict) -> "MinedResult":
        pass

    def prettify(self) -> str:
        try:
            typ_string = self.expr_type.to_string()
        except:
            import traceback

            print(traceback.format_exc())
            typ_string = self.expr_type.type_json

        with contextlib.redirect_stdout(io.StringIO()) as f_out:
            print(f"UID: {self.uid}\nKind: {self.kind}\nReference: {self.reference}")
            print("----------")
            print(f"Code:\n{self.code}")
            print("----------")
            print(f"Templatized:\n{self.template}")
            print("----------")
            print(f"Extra Context Vars:\n{self.extra_context_vars}")
            print("----------")
            print(f"Value Type: {typ_string}")
            print("==========")

        return f_out.getvalue()

    def __repr__(self):
        return self.prettify()

    def __str__(self):
        return self.prettify()


def is_purely_df_or_series_like(expr_type: SerializedMypyType):
    if not (expr_type.equals(DF_TYPE) or expr_type.equals(SERIES_TYPE)):
        return False

    if expr_type.is_union_type():
        return all(
            is_purely_df_or_series_like(i) or i.is_any_type()
            for i in expr_type.unpack_union_type()
        )
    else:
        return True


def find_library_usages(code_ast: astlib.AstNode) -> Dict[astlib.Name, str]:
    """Finds variable uses that correspond to imports / library usage"""
    #  TODO: Perform proper dataflow analysis (reaching defs)
    result: Dict[astlib.Name, str] = {}
    defs, accesses = astlib.get_definitions_and_accesses(code_ast)
    for def_ in defs:
        if def_.enclosing_node is not None and isinstance(
            def_.enclosing_node, (astlib.Import, astlib.ImportFrom)
        ):
            key_dict = {}
            if isinstance(def_.enclosing_node, astlib.Import):
                prefix = ""

            elif (
                isinstance(def_.enclosing_node, astlib.ImportFrom)
                and def_.enclosing_node.module is not None
            ):
                prefix = astlib.to_code(def_.enclosing_node.module).strip() + "."

            else:
                continue

            for alias in def_.enclosing_node.names:
                name_str = astlib.to_code(alias.name).strip()
                if alias.asname is None:
                    key_dict[name_str] = f"{prefix}{name_str}"
                else:
                    key_dict[
                        astlib.to_code(alias.asname.name).strip()
                    ] = f"{prefix}{name_str}"

            for access in accesses:
                if isinstance(access.node, astlib.Name):
                    if access.node.value in key_dict:
                        result[access.node] = key_dict[access.node.value]

    return result


def find_constants(code_ast: astlib.AstNode) -> Dict[astlib.BaseExpression, Any]:
    """Finds constant expressions in the AST. Sound but not necessarily complete right now."""
    #  TODO: Perform proper dataflow analysis (constant propagation)
    result: Dict[astlib.BaseExpression, Any] = {}
    defs, accesses = astlib.get_definitions_and_accesses(code_ast)

    #  We will only focus on accesses whose defs are top-level statements to avoid
    #  having to bother about loops etc.
    top_level_stmts = set(astlib.iter_body_stmts(code_ast))
    accesses = [
        a
        for a in accesses
        if all(d.enclosing_node in top_level_stmts for d in a.definitions)
    ]

    numbering: Dict[astlib.AstNode, int] = {}
    for idx, stmt in enumerate(astlib.iter_body_stmts(code_ast)):
        for node in astlib.walk(stmt):
            numbering[node] = idx

    forbidden_access_nodes: Set[astlib.Name] = set()
    for node in astlib.walk(code_ast):
        #  Do not want even a hint of an update.
        if isinstance(node, (astlib.Assign, astlib.AugAssign, astlib.AnnAssign)):
            targets: List[astlib.AstNode] = []
            if isinstance(node, astlib.Assign):
                targets = list(node.targets)
            elif isinstance(node, astlib.AugAssign):
                targets = [node.target]
            elif isinstance(node, astlib.AnnAssign):
                targets = [node.target]

            for target in targets:
                for c_node in astlib.walk(target):
                    if isinstance(c_node, astlib.Name):
                        # print(f"Adding {astlib.to_code(c_node)} for {astlib.to_code(node)}")
                        forbidden_access_nodes.add(c_node)

        #  Any sort of call on a name could mean a side-effect. Just avoid.
        elif isinstance(node, astlib.Call):
            for c_node in astlib.walk(node.func):
                if isinstance(c_node, astlib.Name):
                    # print(f"Adding {astlib.to_code(c_node)} for {astlib.to_code(node)}")
                    forbidden_access_nodes.add(c_node)

    for access in accesses:
        num = numbering[access.node]
        #  Find the closest top-level def
        cur, score = None, None
        for def_ in access.definitions:
            d_num = numbering[def_.enclosing_node]
            if d_num < num and (score is None or d_num > score):
                cur, score = def_, d_num

        if cur is None:
            continue

        if not isinstance(cur.enclosing_node, (astlib.AnnAssign, astlib.Assign)):
            continue

        if any(acc.node in forbidden_access_nodes for acc in cur.accesses):
            continue

        if astlib.is_constant(cur.enclosing_node.value):
            val = astlib.get_constant_value(cur.enclosing_node.value)
            result[access.node] = val

    return result


def find_single_use_expressions(
    code_ast: astlib.AstNode,
) -> Dict[astlib.BaseExpression, Any]:
    """Finds single-use (one def and one use). Sound but not necessarily complete right now."""
    #  TODO: Perform proper dataflow analysis (constant propagation)
    result: Dict[astlib.BaseExpression, Any] = {}
    defs, accesses = astlib.get_definitions_and_accesses(code_ast)

    #  We will only focus on accesses whose defs are top-level statements to avoid
    #  having to bother about loops etc.
    top_level_stmts = set(astlib.iter_body_stmts(code_ast))
    accesses = [
        a
        for a in accesses
        if all(d.enclosing_node in top_level_stmts for d in a.definitions)
    ]

    for access in accesses:
        if len(access.definitions) != 1:
            continue

        cur = access.definitions[0]
        if len(cur.accesses) != 1:
            continue

        if not isinstance(cur.enclosing_node, (astlib.AnnAssign, astlib.Assign)):
            continue

        print("OKAY", astlib.to_code(cur.enclosing_node), astlib.to_code(access.node))

    return result


def replace_constants(
    target: astlib.AstNode,
    true_exprs: Collection[astlib.BaseExpression],
    free_vars: Collection[astlib.Name],
    constants: Dict[astlib.BaseExpression, Any],
) -> Tuple[NewTarget, NodeReplMap]:
    """Replace any constant variables with their concrete values, and update the inferred types dict"""
    repl_dict = {}
    for node in true_exprs:
        if (not isinstance(node, astlib.Name)) or node not in free_vars:
            continue

        if node in constants:
            repl_dict[node] = astlib.parse_expr(repr(constants[node]))

    if len(repl_dict) == 0:
        return target, {n: n for n in astlib.walk(target)}

    output_mapping = {}
    target = astlib.with_deep_replacements(target, repl_dict, output_mapping)
    return target, output_mapping


def extract_context(
    target: astlib.AstNode,
    free_vars: Collection[astlib.Name],
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
    lib_usages: Dict[astlib.Name, str],
) -> Optional[Dict]:
    """Checks if there are any undefined variables that are not library usages and not dfs/series. If yes, they
    become part of the context vars. If they do not have a type associated, then we discard the whole snippet."""
    context_vars: Dict[str, Any] = {}
    for node in free_vars:
        if node not in lib_usages:
            if node not in inferred_types:
                return None

            typ = inferred_types[node]
            is_builtin_func = typ.is_callable_type() and node.value in _BUILTIN_FUNCS
            if not (
                typ.equals(DF_TYPE)
                or typ.equals(SERIES_TYPE)
                or typ.is_bool_type()
                or is_builtin_func
            ):
                if typ.is_any_type():
                    return None

                if typ.is_callable_type():
                    return None

                context_vars[node.value] = typ.to_string()

    return context_vars


def normalize_df_series_vars(
    target: astlib.AstNode,
    true_exprs: Collection[astlib.BaseExpression],
    free_vars: Collection[astlib.Name],
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
) -> Tuple[NewTarget, DfArgs, SeriesArgs, NodeReplMap]:
    """Replaces variables corresponding to dataframes or series with standard names"""
    seen_dfs: Dict[str, int] = {}
    df_repl_map: Dict[astlib.Name, astlib.Name] = {}

    seen_series: Dict[str, int] = {}
    series_repl_map: Dict[astlib.Name, astlib.Name] = {}

    for node in true_exprs:
        if (
            (not isinstance(node, astlib.Name))
            or node not in inferred_types
            or node not in free_vars
        ):
            continue

        #  NOTE: If there is a union type of DataFrame and Series, DataFrame will be picked.
        if inferred_types[node].equals(DF_TYPE):
            if node.value not in seen_dfs:
                seen_dfs[node.value] = len(seen_dfs) + 1

            df_repl_map[node] = node  # Will update later

        elif inferred_types[node].equals(SERIES_TYPE):
            if node.value not in seen_series:
                seen_series[node.value] = len(seen_series) + 1

            series_repl_map[node] = node  # Will update later

    if len({i.value for i in df_repl_map.keys()}) <= 1:

        def df_arg_creator(ctr: int):
            return "df"

    else:

        def df_arg_creator(ctr: int):
            return f"df{ctr}"

    if len({i.value for i in series_repl_map.keys()}) <= 1:

        def series_arg_creator(ctr: int):
            return "series"

    else:

        def series_arg_creator(ctr: int):
            return f"series{ctr}"

    for node in df_repl_map.keys():
        df_repl_map[node] = astlib.create_name_expr(
            df_arg_creator(seen_dfs[node.value])
        )

    for node in series_repl_map.keys():
        series_repl_map[node] = astlib.create_name_expr(
            series_arg_creator(seen_series[node.value])
        )

    output_map: NodeReplMap = {}
    target = astlib.with_deep_replacements(
        target, {**df_repl_map, **series_repl_map}, output_map
    )
    return (
        target,
        sorted(i.value for i in df_repl_map.values()),
        sorted(i.value for i in series_repl_map.values()),
        output_map,
    )


def normalize_call_args(
    target: astlib.AstNode,
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
) -> Tuple[NewTarget, NodeReplMap]:
    """Normalize order of keyword arguments"""
    repl_map: NodeReplMap = {}
    for node in astlib.walk(target):
        if not isinstance(node, astlib.Call):
            continue

        call_expr = node
        if (call_expr.func not in inferred_types) or (
            not inferred_types[call_expr.func].is_callable_type()
        ):
            continue

        if any(arg.star != "" for arg in call_expr.args):
            #  TODO: How to handle starred args?
            continue

        pos_args = [arg for arg in call_expr.args if arg.keyword is None]
        kw_args = [arg for arg in call_expr.args if arg.keyword is not None]
        arg_order = inferred_types[call_expr.func].get_callable_arg_order()

        new_args = [*pos_args] + sorted(
            kw_args, key=lambda x: arg_order.get(x.keyword.value, 0)
        )
        if len(new_args) > 0:
            new_args[-1] = new_args[-1].with_changes(
                comma=astlib.cst.MaybeSentinel.DEFAULT
            )

        if new_args != call_expr.args:
            repl_map[call_expr] = call_expr.with_changes(args=new_args)

    output_mapping: NodeReplMap = {}
    if len(repl_map) != 0:
        target = astlib.with_deep_replacements(target, repl_map, output_mapping)

    return target, output_mapping


def normalize_col_accesses(
    target: astlib.AstNode,
    true_exprs: Collection[astlib.BaseExpression],
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
) -> Tuple[NewTarget, NodeReplMap]:
    """Normalizes col accesses by converting attribute-based accesses like df.Price to
    subscript-based such as df['Price']"""
    repl_map: NodeReplMap = {}
    for expr in true_exprs:
        if expr not in inferred_types:
            continue

        expr_typ = inferred_types[expr]
        if isinstance(expr, astlib.Attribute):
            value = expr.value
            if value not in inferred_types:
                continue

            val_typ = inferred_types[value]
            okay = False
            # print("GOT HERE", val_typ, expr_typ)
            if val_typ.equals(DF_TYPE) and (
                expr_typ.equals(DF_TYPE) or expr_typ.equals(SERIES_TYPE)
            ):
                try:
                    if (not hasattr(pd.DataFrame, expr.attr.value)) and (
                        not hasattr(pd.Series, expr.attr.value)
                    ):
                        okay = True
                except:
                    pass
            elif val_typ.equals(DF_GROUPBY_TYPE) and (
                expr_typ.equals(DF_GROUPBY_TYPE) or expr_typ.equals(SERIES_GROUPBY_TYPE)
            ):
                try:
                    if not hasattr(
                        pd.core.groupby.generic.DataFrameGroupBy, expr.attr.value
                    ):
                        okay = True
                except:
                    pass

            if okay:
                new_node = astlib.parse_expr(
                    f'dummy["{expr.attr.value}"]'
                ).with_changes(value=expr.value)
                repl_map[expr] = new_node

    output_mapping: NodeReplMap = {}
    if len(repl_map) != 0:
        cur_code: str = astlib.to_code(target)
        changed: bool = True
        while changed:
            changed = False
            target = astlib.with_deep_replacements(target, repl_map, output_mapping)
            new_code = astlib.to_code(target)
            if cur_code != new_code:
                changed = True
                cur_code = new_code

    return target, output_mapping


def templatize(
    target: astlib.AstNode,
    true_exprs: Collection[astlib.BaseExpression],
    free_vars: Collection[astlib.Name],
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType],
    lib_usages: Dict[astlib.Name, str],
) -> Tuple[NewTarget, Dict[str, List[str]]]:
    """Replace constants and remaining variable names with standard ones to create a template suitable for clustering"""
    type_to_exprs: Dict[str, List[astlib.BaseExpression]] = collections.defaultdict(
        list
    )
    allowed_key_chars = set(string.ascii_letters + string.digits + "_")
    for node in true_exprs:
        is_const = astlib.is_constant(node)
        const_val = None if not is_const else astlib.get_constant_value(node)
        if not ((isinstance(node, astlib.Name) and node in free_vars) or is_const):
            continue

        if node in lib_usages:
            continue

        if node not in inferred_types:
            if not is_const:
                continue

            key = type(const_val).__name__

        else:
            typ = inferred_types[node]
            if typ.equals(DF_TYPE):
                key = "df"
            elif typ.equals(SERIES_TYPE):
                key = "series"
            elif typ.is_callable_type():
                continue
            elif typ.is_str_type():
                key = "str"
            elif typ.is_int_type():
                key = "int"
            elif typ.is_bool_type():
                if isinstance(node, astlib.Name) and node.value in {"True", "False"}:
                    continue
                key = "bool"
            elif typ.is_float_type():
                key = "float"
            elif is_const and isinstance(const_val, (set, dict, list, tuple)):
                if len(const_val) == 0:
                    continue

                if isinstance(const_val, (set, list, tuple)):
                    first_elem_class = type(next(iter(const_val)))
                    if all(isinstance(elem, first_elem_class) for elem in const_val):
                        key = first_elem_class.__name__
                    else:
                        key = ""

                    if isinstance(const_val, set):
                        key += "_set"
                    elif isinstance(const_val, list):
                        key += "_list"
                    elif isinstance(const_val, tuple):
                        key += "_tuple"

                else:
                    key = "dict"

            else:
                while typ.is_union_type():
                    typ = typ.unpack_union_type()[0]

                if isinstance(typ.type_json, str):
                    key = typ.type_json
                else:
                    key = str(typ.type_json.get(".class", "VAR"))

        key = "".join(i if i in allowed_key_chars else "_" for i in key)
        type_to_exprs[key].append(node)
        # print("Adding", key, astlib.to_code(node))

    ctr_map: Dict[str, Dict[str, int]] = {k: {} for k in type_to_exprs.keys()}
    repl_map: NodeReplMap = {}
    names_map: Dict[str, List[str]] = collections.defaultdict(list)
    for typ_key, exprs in type_to_exprs.items():
        ctr_map_entry = ctr_map[typ_key]
        for expr in exprs:
            node_key = astlib.to_code(expr)
            if node_key not in ctr_map_entry:
                ctr_map_entry[node_key] = idx = len(ctr_map_entry) + 1
                names_map[typ_key].append(f"{typ_key.upper()}{idx}")

            idx = ctr_map_entry[node_key]
            repl_map[expr] = astlib.create_name_expr(f"{typ_key.upper()}{idx}")

    return astlib.with_deep_replacements(target, repl_map), names_map


def get_mypy_cache_dir_path(uid: int) -> str:
    """Returns a cache dir to use for mypy based on a UID. Useful for multiprocess safety."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(script_dir, f".mypy_cache{uid}")


def get_created_mypy_cache_dir_paths() -> List[str]:
    """Returns all the created mypy cache dirs"""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    return glob.glob(os.path.join(script_dir, ".mypy_cache*"))
