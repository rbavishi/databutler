import collections
import os
import shlex
from typing import Deque, Tuple, Dict, Any, Optional, List, Set, Type

import attrs

import pandas as pd
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import code as codeutils


@attrs.define(eq=False, repr=False)
class AutodocFewShotExample:
    code: str
    nl: str
    param_nl: str
    param_code: str

    @classmethod
    def from_json(cls, val_json: Dict) -> "AutodocFewShotExample":
        return AutodocFewShotExample(
            code=val_json["code"],
            nl=val_json["nl"],
            param_nl=val_json["param_nl"],
            param_code=val_json["param_code"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "nl": self.nl,
            "param_nl": self.param_nl,
            "param_code": self.param_code,
        }


@attrs.define(eq=False, repr=False)
class Parameterization:
    nl: str
    code: str

    all_params: List[str]
    df_params: List[str]
    series_params: List[str]
    col_params: List[str]

    instantiation: Dict[str, str]

    def get_instantiated_nl(self) -> str:
        nl = self.nl
        for k, v in self.instantiation.items():
            nl = nl.replace(f"[{k}]", v)

        return nl

    def get_instantiated_code(self) -> str:
        code = self.code
        for k, v in sorted(
            self.instantiation.items(), key=lambda x: len(x[0]), reverse=True
        ):
            code = code.replace(f"{k}", v)

        return code


@attrs.define(eq=False, repr=False)
class AutodocDescription:
    uid: str
    success: bool
    nl: str
    generated_code: str
    code_parseable: bool
    assistance_level: int

    nl_based_parameterization: Parameterization = None
    llm_based_parameterization: Parameterization = None
    is_derived: bool = False


@attrs.define(eq=False, repr=False)
class AutodocResult:
    uid: str
    success: bool
    ground_truth_code: str

    correct_descriptions: List[AutodocDescription]
    incorrect_descriptions: List[AutodocDescription]


def normalize_keyword_argument_order(code: str):
    code_ast = astlib.parse(code)
    #  Normalize keyword argument order
    node_repl = {}
    for node in astlib.walk(code_ast):
        if isinstance(node, astlib.Call):
            pos_args = [arg for arg in node.args if arg.keyword is None]
            kw_args = sorted(
                [
                    astlib.with_changes(arg, comma=astlib.cst.MaybeSentinel.DEFAULT)
                    for arg in node.args
                    if arg.keyword is not None
                ],
                key=lambda x: x.keyword.value,
            )
            if len(kw_args) > 0:
                node_repl[node] = astlib.with_changes(node, args=pos_args + kw_args)

    if len(node_repl) > 0:
        code_ast = astlib.with_deep_replacements(code_ast, node_repl)
        return codeutils.normalize_code_fast(astlib.to_code(code_ast))
    else:
        return codeutils.normalize_code_fast(code)


def normalize_code_for_comparison(
    code: str, df_args: Set[str], replace_singleton_lists: bool = True
):
    code = normalize_keyword_argument_order(code)
    code_ast = astlib.parse(code)

    repl_map = {}

    #  Replace attribute-based column access to the best of our ability
    if len(df_args) == 0:
        for node in astlib.walk(code_ast):
            if (
                isinstance(node, astlib.Attribute)
                and isinstance(node.value, astlib.Name)
                and node.value.value in df_args
            ):
                if not hasattr(pd.DataFrame, node.attr.value):
                    new_node = astlib.parse_expr(
                        f'{node.value.value}["{node.attr.value}"]'
                    )
                    repl_map[node] = new_node

    if replace_singleton_lists:
        #  Replace singleton lists with the element. This is really useful for pandas where a column or a singleton
        #  list containing a column often have the same meaning.
        for node in astlib.walk(code_ast):
            if isinstance(node, astlib.List):
                if len(node.elements) == 1 and isinstance(
                    node.elements[0], astlib.cst.Element
                ):
                    repl_map[node] = node.elements[0].value

    if len(repl_map) > 0:
        code_ast = astlib.with_deep_replacements(code_ast, repl_map)

    return codeutils.normalize_code_fast(astlib.to_code(code_ast))


def get_few_shot_example_path(campaign_dir: str, version: int) -> str:
    return os.path.join(campaign_dir, f"few_shot_{version}.yaml")


def quotes_aware_split_with_whitespace(text: str) -> List[str]:
    #  Shlex helps the splitting be quotes-aware
    text_tokens = shlex.split(text, posix=False)
    text_tokens_with_whitespace = []
    idx = 0
    cur_token_idx = 0
    cur_whitespace_tokens = []
    while idx < len(text):
        if text[idx] == text_tokens[cur_token_idx][0]:
            if len(cur_whitespace_tokens) != 0:
                text_tokens_with_whitespace.append("".join(cur_whitespace_tokens))
                cur_whitespace_tokens.clear()

            idx += len(text_tokens[cur_token_idx])
            text_tokens_with_whitespace.append(text_tokens[cur_token_idx])
            cur_token_idx += 1
        else:
            cur_whitespace_tokens.append(text[idx])
            idx += 1

    if len(cur_whitespace_tokens) != 0:
        text_tokens_with_whitespace.append("".join(cur_whitespace_tokens))

    return text_tokens_with_whitespace


def find_instantiation_map(template_ast: astlib.AstNode, code_ast: astlib.AstNode):
    worklist: Deque[Tuple[astlib.AstNode, astlib.AstNode]] = collections.deque()
    worklist.append((template_ast, code_ast))

    req_mapping = {}

    while len(worklist) > 0:
        t_node, c_node = worklist.popleft()

        if not isinstance(t_node, type(c_node)):
            req_mapping[t_node] = c_node
            continue

        t_children = t_node.children
        c_children = c_node.children

        if len(t_children) != len(c_children):
            req_mapping[t_node] = c_node
            continue

        if len(t_children) == 0 and not t_node.deep_equals(c_node):
            # if astlib.to_code(t_node) != "":
            req_mapping[t_node] = c_node
            continue

        for t_c, c_c in zip(t_children, c_children):
            worklist.append((t_c, c_c))

    return req_mapping


def generate_llm_based_parameterization(
    desc: AutodocDescription,
    param_nl: str,
    param_code: str,
    orig_code: str,
) -> Optional[Parameterization]:
    try:
        #  It must be parseable
        code_ast = astlib.parse(param_code)
    except:
        return None

    first_mod_stmt = next(iter(astlib.iter_body_stmts(code_ast)))
    #  It should be a function
    if not isinstance(first_mod_stmt, astlib.FunctionDef):
        return None

    #  With a single statement as the body
    if len(list(astlib.iter_body_stmts(first_mod_stmt))) != 1:
        return None

    first_func_stmt = next(iter(astlib.iter_body_stmts(first_mod_stmt)))
    #  It should be a return statement or a single expression statement
    if not isinstance(first_func_stmt, astlib.Return) and not isinstance(
        first_func_stmt, astlib.Expr
    ):
        return None
    else:
        param_expr = first_func_stmt.value

    params: List[str] = [param.name.value for param in first_mod_stmt.params.params]
    if not all(f"[{p}]" in param_nl for p in params):
        #  Every parameter must be mentioned explicitly in the parameterized description
        return None

    #  There must be a valid instantiation to the original code
    param_expr = astlib.parse_expr(
        normalize_keyword_argument_order(astlib.to_code(param_expr))
    )
    orig_code = normalize_keyword_argument_order(orig_code)
    inst_map = find_instantiation_map(param_expr, astlib.parse_expr(orig_code))
    if not all(
        isinstance(k, astlib.Name) and k.value in params for k in inst_map.keys()
    ):
        return None
    else:
        instantiation = {
            k.value: astlib.to_code(v)
            for k, v in inst_map.items()
            if isinstance(k, astlib.Name)
        }

    return Parameterization(
        nl=param_nl,
        code=param_code,
        df_params=[p for p in params if p.startswith("df")],
        series_params=[p for p in params if p.startswith("series")],
        col_params=[p for p in params if p.startswith("col")],
        all_params=params,
        instantiation=instantiation,
    )


def generate_nl_based_parameterization(
    snippet: MinedResult, generated_nl: str
) -> Optional[Parameterization]:
    try:
        nl_tokens = quotes_aware_split_with_whitespace(generated_nl)
    except:
        return None

    nl_tokens_set = set(nl_tokens)

    nl_repl_dict: Dict[str, str] = {}
    node_repl_dict: Dict[astlib.AstNode, astlib.AstNode] = {}
    code = snippet.code
    code_ast = astlib.parse(snippet.code)
    type_ctrs = collections.Counter()

    df_params: List[str] = []
    series_params: List[str] = []
    col_params: List[str] = []
    const_params: Dict[str, Type] = {}
    instantiation: Dict[str, str] = {}

    def get_new_param_name(val: Any, is_col: bool = False):
        val_type_name = type(val).__name__
        type_ctrs[val_type_name, is_col] += 1
        return f"PARAM{'_COL' if is_col else ''}_{val_type_name}{type_ctrs[val_type_name, is_col]}"

    for node in astlib.walk(code_ast):
        if astlib.is_constant(node):
            val = astlib.get_constant_value(node)
            to_replace: Optional[str] = None
            is_col: bool = False
            if isinstance(val, str):
                if f'"COL:{val}"' in nl_tokens_set:
                    to_replace = f'"COL:{val}"'
                    is_col = True
                elif f"'COL:{val}'" in nl_tokens_set:
                    to_replace = f"'COL:{val}'"
                    is_col = True
                elif f'"{val}"' in nl_tokens_set:
                    to_replace = f'"{val}"'
                elif f"'{val}'" in nl_tokens_set:
                    to_replace = f"'{val}'"

            elif repr(val) in nl_tokens_set:
                to_replace = repr(val)

            elif (
                isinstance(val, (int, float))
                and val < 0
                and f"-{abs(val)}" in nl_tokens_set
            ):
                #  Negative numbers are sometimes represented weirdly in normalized code.
                to_replace = f"-{abs(val)}"

            elif (
                f"'{val!r}'" in nl_tokens_set
                and f'"{val!r}"' not in code
                and f"'{val!r}'" not in code
            ):
                to_replace = f"'{val!r}'"

            elif (
                f'"{val!r}"' in nl_tokens_set
                and f'"{val!r}"' not in code
                and f"'{val!r}'" not in code
            ):
                to_replace = f'"{val!r}"'

            if to_replace is not None:
                if to_replace not in nl_repl_dict:
                    new_name = get_new_param_name(val, is_col=is_col)
                    nl_repl_dict[to_replace] = f"[{new_name}]"
                else:
                    new_name = nl_repl_dict[to_replace][1:-1]

                node_repl_dict[node] = astlib.create_name_expr(new_name)
                const_params[new_name] = type(val)
                if is_col and new_name not in col_params:
                    col_params.append(new_name)

                instantiation[new_name] = astlib.to_code(node)

        elif isinstance(node, astlib.Name):
            if node.value in snippet.df_vars:
                new_name = f"PARAM_DF{node.value[len('df'):]}"
                nl_repl_dict[f'"{node.value}"'] = f"[{new_name}]"
                node_repl_dict[node] = astlib.create_name_expr(new_name)
                df_params.append(new_name)
                instantiation[new_name] = astlib.to_code(node)

            elif node.value in snippet.series_vars:
                new_name = f"PARAM_SERIES{node.value[len('ss'):]}"
                nl_repl_dict[f'"{node.value}"'] = f"[{new_name}]"
                node_repl_dict[node] = astlib.create_name_expr(new_name)
                series_params.append(new_name)
                instantiation[new_name] = astlib.to_code(node)

    new_tokens = [nl_repl_dict.get(tok, tok) for tok in nl_tokens]
    new_nl = "".join(new_tokens)

    new_code = codeutils.normalize_code_fast(
        astlib.to_code(astlib.with_deep_replacements(code_ast, node_repl_dict))
    )

    if '"COL:' in new_nl or "'COL:" in new_nl:
        return None

    # print("NEW GENERATED PARAMETERIZATION", new_nl, "||", new_code)
    print(instantiation)

    return Parameterization(
        nl=new_nl,
        code=new_code,
        df_params=df_params,
        series_params=series_params,
        col_params=col_params,
        all_params=list(set(df_params) | set(series_params) | set(const_params.keys())),
        instantiation=instantiation,
    )


def attempt_parameterization_application(
    parameterization: Parameterization, snippet: MinedResult
) -> Optional[Parameterization]:
    if parameterization.code.strip().startswith("def code"):
        #  The template is in function form. Extract the return value or last expression statement value.
        template_ast = astlib.parse(parameterization.code)
        first_stmt = next(astlib.iter_body_stmts(template_ast))
        if not isinstance(first_stmt, astlib.FunctionDef):
            return None
        first_func_stmt = next(astlib.iter_body_stmts(first_stmt))
        if not (
            isinstance(first_func_stmt, astlib.Expr)
            or isinstance(first_func_stmt, astlib.Return)
        ):
            return None

        template_ast = first_func_stmt.value
    else:
        template_ast = astlib.parse_expr(parameterization.code)

    template_ast = astlib.parse_expr(
        normalize_keyword_argument_order(astlib.to_code(template_ast))
    )
    code_ast = astlib.parse_expr(normalize_keyword_argument_order(snippet.code))

    worklist: Deque[Tuple[astlib.AstNode, astlib.AstNode]] = collections.deque()
    worklist.append((template_ast, code_ast))

    all_params: Set[str] = set(parameterization.all_params)
    df_and_series_params: Set[str] = set(parameterization.df_params) | set(
        parameterization.series_params
    )
    df_and_series_vars = set(snippet.df_vars) | set(snippet.series_vars)
    instantiation: Dict[str, str] = {}

    while len(worklist) > 0:
        t_node, c_node = worklist.popleft()

        if isinstance(t_node, astlib.Name) and t_node.value in all_params:
            if t_node.value in df_and_series_params:
                if (
                    isinstance(c_node, astlib.Name)
                    and c_node.value in df_and_series_vars
                ):
                    if t_node.value in instantiation:
                        if instantiation[t_node.value] != c_node.value:
                            return None
                    else:
                        instantiation[t_node.value] = c_node.value
                else:
                    return None

            elif astlib.is_constant(c_node):
                try:
                    t_val = eval(parameterization.instantiation[t_node.value])
                except:
                    return None
                else:
                    val = astlib.get_constant_value(c_node)
                    if type(val) == type(t_val):
                        val_repr = astlib.to_code(c_node)
                        if t_node.value in instantiation:
                            if instantiation[t_node.value] != val_repr:
                                return None
                        else:
                            instantiation[t_node.value] = val_repr
                    else:
                        return None
            else:
                return None

        else:
            if not isinstance(t_node, type(c_node)):
                return None

            t_children = t_node.children
            c_children = c_node.children

            if len(t_children) != len(c_children):
                return None

            if len(t_children) == 0 and not t_node.deep_equals(c_node):
                return None

            for t_c, c_c in zip(t_children, c_children):
                worklist.append((t_c, c_c))

    if set(instantiation.keys()) != all_params:
        return None

    return Parameterization(
        nl=parameterization.nl,
        code=parameterization.code,
        df_params=parameterization.df_params,
        series_params=parameterization.series_params,
        col_params=parameterization.col_params,
        all_params=parameterization.all_params,
        instantiation=instantiation,
    )
