import contextlib
import io
import os
import re
import tempfile
from typing import Dict, Tuple, Optional, Set, Union, List

from databutler.pat import astlib
from databutler.pat.analysis.type_analysis.mypy_types import SerializedMypyType
from databutler.utils import code as codeutils

STUBS_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "stubs")


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class _TypeInferenceInstrumenter(astlib.AstTransformer):
    def __init__(self):
        super().__init__()
        self._node_to_idx_mapping: Dict[astlib.BaseExpression, int] = {}
        self._cur_ast: Optional[astlib.AstNode] = None
        self._uninstrumentable_exprs: Set[astlib.AstNode] = set()

    def process(self, ast_root: astlib.AstNode) -> Tuple[astlib.AstNode, Dict[astlib.BaseExpression, int]]:
        self._node_to_idx_mapping.clear()
        self._cur_ast = ast_root
        self._uninstrumentable_exprs.clear()
        return ast_root.visit(self), self._node_to_idx_mapping.copy()

    def on_visit(self, node: astlib.AstNode):
        if isinstance(node, astlib.ConcatenatedString):
            self._uninstrumentable_exprs.add(node.left)
            self._uninstrumentable_exprs.add(node.right)

        if astlib.is_starred_expr(node):
            self._uninstrumentable_exprs.add(node)

        if isinstance(node, astlib.Decorator):
            self._uninstrumentable_exprs.add(node)
            return False

        return True

    def on_leave(self, original_node: astlib.AstNode, updated_node: astlib.AstNode):
        if isinstance(original_node, astlib.BaseExpression):
            assert isinstance(updated_node, astlib.BaseExpression)
            if astlib.expr_is_evaluated(original_node, context=self._cur_ast):
                if original_node not in self._uninstrumentable_exprs:
                    return self._process_expr(original_node, updated_node)

        return updated_node

    def _add_parens(self, node: astlib.BaseExpression) -> astlib.BaseExpression:
        return node.with_changes(
            lpar=[astlib.cst.LeftParen()],
            rpar=[astlib.cst.RightParen()],
        )

    def _process_expr(
            self, original_node: astlib.BaseExpression, updated_node: astlib.BaseExpression
    ) -> astlib.BaseExpression:
        if original_node not in self._node_to_idx_mapping:
            self._node_to_idx_mapping[original_node] = len(self._node_to_idx_mapping)

        idx = self._node_to_idx_mapping[original_node]

        if isinstance(updated_node, astlib.FormattedString):
            updated_node = updated_node.with_changes(start='f"""', end='"""')

        tuple_expr = astlib.create_tuple_expr([
            self._add_parens(updated_node),
            self._create_reveal_type_call(astlib.SimpleString(repr(f"_TYPE_IDX_{idx}"))),
            self._create_reveal_type_call(self._add_parens(original_node)),
        ])
        index_expr = astlib.parse_expr("dummy[2]")
        new_node = astlib.with_changes(index_expr, value=tuple_expr)

        return new_node

    def _create_reveal_type_call(self, expr: astlib.BaseExpression) -> astlib.Call:
        return astlib.Call(func=astlib.create_name_expr("reveal_type"), args=[astlib.Arg(value=expr)])


def capture_revealed_types(type_store: List[SerializedMypyType]) -> None:
    import mypy.messages

    class NewMessageBuilder(mypy.messages.MessageBuilder):
        def reveal_type(self, typ, context) -> None:
            type_store.append(SerializedMypyType.from_mypy_type(typ))
            # print("TYPE", typ.serialize())
            # print("CONTEXT", context, id(context))
            super().reveal_type(typ, context)

    mypy.messages.MessageBuilder = NewMessageBuilder


def run_mypy(
        source: Union[str, astlib.AstNode],
        cache_dir: Optional[str] = None,
) -> Tuple[astlib.AstNode, Dict[astlib.BaseExpression, SerializedMypyType]]:
    if isinstance(source, str):
        src_ast = astlib.parse(source)
    elif isinstance(source, astlib.AstNode):
        src_ast = source
    else:
        raise TypeError("Source must be a string or an AST")

    inst_ast, node_to_idx = _TypeInferenceInstrumenter().process(src_ast)
    inst_src = codeutils.unparse_astlib_ast(inst_ast)
    # print(inst_src)
    inferred_types: Dict[astlib.BaseExpression, SerializedMypyType] = {}
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    collected_types: List[SerializedMypyType] = []
    capture_revealed_types(collected_types)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    type_idx_regex = re.compile(r'_TYPE_IDX_([0-9]*)')

    with tempfile.NamedTemporaryFile(mode='w') as fp:
        fp.write(codeutils.normalize_code(inst_src))
        fp.flush()

        try:
            with set_env(MYPYPATH=STUBS_PATH):
                from mypy.main import main
                args = [fp.name, "--ignore-missing-imports"]
                if cache_dir is not None:
                    args.append(f"--cache-dir={cache_dir}")

                main(None,
                     args=args,
                     stdout=stdout_buf, stderr=stderr_buf,
                     clean_exit=True)
        except BaseException as e:
            # print(e)
            pass

        last_idx: Optional[int] = None
        for typ in collected_types:
            if typ.is_literal():
                val = typ.get_literal_value()
                if isinstance(val, str) and val.startswith("_TYPE_IDX"):
                    idx = int(type_idx_regex.search(val).group(1))
                    last_idx = idx
                    continue

            if last_idx is not None:
                inferred_types[idx_to_node[last_idx]] = typ

    # for node, inferred_type in inferred_types.items():
    #     print(astlib.to_code(node), "   :    ", inferred_type)

    # print(stdout_buf.getvalue())
    # print(stderr_buf.getvalue())
    return src_ast, inferred_types
