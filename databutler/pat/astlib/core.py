import collections
import itertools
import weakref
#  We use the CSTNode as the default AST node.
from typing import MutableMapping, Iterator, Union, Mapping, Optional
import typing

import attr
import libcst as cst

from databutler.pat.astlib.expr_contexts import ExpressionContextProvider, ExpressionContext
from databutler.pat.astlib.position import NodePosition
from libcst import AnnAssign, Assert, Assign, Attribute, AugAssign, Await, BaseAssignTargetExpression, BaseComp, \
    BaseCompoundStatement, BaseDelTargetExpression, BaseDict, BaseExpression, BaseList, BaseNumber, BaseSet, \
    BaseSimpleComp, BaseSmallStatement, BaseString, BinaryOperation, BooleanOperation, Break, Call, ClassDef, \
    Comparison, ConcatenatedString, Continue, Del, Dict, DictComp, Ellipsis, Expr, Float, For, FormattedString, \
    FunctionDef, GeneratorExp, Global, If, IfExp, Imaginary, Import, ImportFrom, Integer, Lambda, List, ListComp, \
    Name, NamedExpr, Nonlocal, Pass, Raise, Return, Set, SetComp, SimpleString, Subscript, Try, Tuple, UnaryOperation, \
    While, With, Yield, Module, Arg, Param, Decorator

from databutler.pat.astlib.editing import ChildReplacementTransformer, StmtRemovalAndSimplificationTransformer, \
    NodeRemovalTransformer
from databutler.pat.astlib.notebooks import NotebookCell, NotebookCellBody, parse_ipynb

AstNode = cst.CSTNode
AstTransformer = cst.CSTTransformer
NodeUID = int
ScopeId = int
AstStatementT = Union[cst.BaseSmallStatement, cst.BaseCompoundStatement]
AstStatement = (cst.BaseSmallStatement, cst.BaseCompoundStatement)
ExprContextEligible = (cst.Attribute, cst.Name, cst.Subscript, cst.StarredElement, cst.List, cst.Tuple,
                       cst.NameItem)

#  We maintain global counters to avoid unintended collisions.
_AST_UID_COUNTER = 0
_AST_SCOPE_ID_COUNTER = 0


def reset_uid_counter(new_init_value: int = 0):
    global _AST_UID_COUNTER
    _AST_UID_COUNTER = new_init_value


def reset_scope_id_counter(new_init_value: int = 0):
    global _AST_SCOPE_ID_COUNTER
    _AST_SCOPE_ID_COUNTER = new_init_value


def get_next_uid():
    global _AST_UID_COUNTER
    result = _AST_UID_COUNTER
    _AST_UID_COUNTER += 1
    return result


def get_next_scope_id():
    global _AST_SCOPE_ID_COUNTER
    result = _AST_SCOPE_ID_COUNTER
    _AST_SCOPE_ID_COUNTER += 1
    return result


def children(node: AstNode) -> Iterator[AstNode]:
    """
    Iterate over the direct children.
    :param node:
    :return:
    """
    yield from node.children


def walk(node: AstNode) -> Iterator[AstNode]:
    """
    Depth-first iteration of the AST
    :param node:
    :return:
    """
    yield node
    for c in node.children:
        yield from walk(c)


@attr.s(repr=False, cmp=False)
class Access:
    node: BaseExpression = attr.ib()
    scope_id: int = attr.ib()
    definitions: typing.List['Definition'] = attr.ib()


@attr.s(repr=False, cmp=False)
class Definition:
    name: str = attr.ib()
    node: AstNode = attr.ib()
    enclosing_node: AstNode = attr.ib()
    scope_id: int = attr.ib()
    accesses: typing.List[Access] = attr.ib()


@attr.s
class _AstMetadata:
    """
    Contains extra meta-data about the AST such as position of tokens, scopes, parent/child info etc.
    """
    _ast: weakref.ReferenceType = attr.ib()
    _uid_dict: typing.Dict[Union[AstNode, int], Union[int, AstNode]] = attr.ib()
    _pos_ranges: Mapping[AstNode, NodePosition] = attr.ib()
    _parent_map: Mapping[AstNode, AstNode] = attr.ib()
    _expr_context: Mapping[AstNode, Optional[ExpressionContext]] = attr.ib()
    _scopes: Mapping[AstNode, cst.metadata.Scope] = attr.ib()
    _scope_id_dict: typing.Dict[Union[cst.metadata.Scope, int], Union[int, cst.metadata.Scope]] = attr.ib()
    _config_for_parsing = attr.ib(default=None)
    _defs_and_accesses: typing.Tuple[typing.List[Definition],
                                     typing.List[Access]] = attr.ib(init=False, default=None)

    @classmethod
    def build(cls, ast_root: AstNode) -> '_AstMetadata':
        #  The unsafe_skip_copy avoids making copies of ast_root.
        #  For this reason, make sure ast_root does not have any duplicate nodes internally.
        wrapper = cst.metadata.MetadataWrapper(ast_root, unsafe_skip_copy=True)

        uid_dict = {get_next_uid(): n for n in walk(ast_root)}
        uid_dict.update({v: k for k, v in uid_dict.items()})
        expr_context = {n: None for n in walk(ast_root)}

        pos_ranges = wrapper.resolve(cst.metadata.PositionProvider)
        parent_map = wrapper.resolve(cst.metadata.ParentNodeProvider)
        expr_context.update(wrapper.resolve(ExpressionContextProvider))
        scopes = dict(wrapper.resolve(cst.metadata.ScopeProvider))
        scope_id_dict = {s: get_next_scope_id() for s in set(scopes.values())}
        scope_id_dict.update({v: k for k, v in scope_id_dict.items()})

        for n in walk(ast_root):
            if n not in scopes:
                par = parent_map.get(n, None)
                while par is not None:
                    if par in scopes:
                        scopes[n] = scopes[par]
                        break

                    par = parent_map.get(par, None)

        if isinstance(ast_root, cst.Module):
            config_for_parsing = ast_root.config_for_parsing
        else:
            config_for_parsing = None

        return _AstMetadata(
            ast=weakref.ref(ast_root),
            uid_dict=uid_dict,
            pos_ranges=pos_ranges,
            parent_map=parent_map,
            expr_context=expr_context,
            config_for_parsing=config_for_parsing,
            scopes=scopes,
            scope_id_dict=scope_id_dict
        )

    def get_uid_for_node(self, node: AstNode) -> int:
        return self._uid_dict[node]

    def get_node_for_uid(self, uid: int) -> AstNode:
        return self._uid_dict[uid]

    def get_parent(self, node: AstNode) -> Optional[AstNode]:
        return self._parent_map.get(node, None)

    def iter_parents(self, node: AstNode) -> Iterator[AstNode]:
        cur = self._parent_map.get(node, None)
        while cur is not None:
            yield cur
            cur = self._parent_map.get(cur, None)

    def get_expr_context(self, node: AstNode) -> Optional[ExpressionContext]:
        return self._expr_context.get(node, None)

    def get_config_for_parsing(self):
        return self._config_for_parsing

    def get_node_uid_mapping(self) -> Mapping[Union[AstNode, int], Union[int, AstNode]]:
        return self._uid_dict

    def get_scope_id_mapping(self) -> Mapping[AstNode, int]:
        return {n: self._scope_id_dict[s] for n, s in self._scopes.items()}

    def get_definitions_and_accesses(self) -> typing.Tuple[typing.List[Definition], typing.List[Access]]:
        if self._defs_and_accesses is not None:
            return self._defs_and_accesses

        all_definitions: typing.Dict = {}
        all_accesses: typing.Dict = {}
        all_scopes = set(self._scopes.values())
        for scope in all_scopes:
            for definition in scope.assignments:
                if not isinstance(definition, cst.metadata.Assignment):
                    continue

                all_definitions[definition] = None
                all_accesses.update({a: None for a in definition.references if not a.is_annotation})

        for definition in all_definitions.keys():
            enclosing_node_types = (cst.BaseSmallStatement, cst.BaseCompoundStatement, cst.CompFor, cst.Lambda)
            if not isinstance(definition.node, enclosing_node_types):
                enclosing_node = next(i for i in self.iter_parents(definition.node)
                                      if isinstance(i, enclosing_node_types))
            else:
                enclosing_node = definition.node

            all_definitions[definition] = Definition(
                name=definition.name,
                node=definition.node,
                enclosing_node=enclosing_node,
                scope_id=self._scope_id_dict[definition.scope],
                accesses=[],
            )

        for access in all_accesses.keys():
            all_accesses[access] = Access(
                node=access.node,
                scope_id=self._scope_id_dict[access.scope],
                definitions=[],
            )

        for k, v in all_definitions.items():
            v.accesses.extend(all_accesses[a] for a in k.references if a in all_accesses)

        for k, v in all_accesses.items():
            v.definitions.extend(all_definitions[a] for a in k.referents if a in all_definitions)

        definitions = list(all_definitions.values())
        accesses = list(all_accesses.values())

        #  Handle AugAssigns with cst.Name as targets specially.
        for n in walk(self._ast()):
            if isinstance(n, AugAssign) and isinstance(n.target, Name):
                scope = self._scopes[n]
                defs = [all_definitions[d] for d in scope[n.target.value]]
                accesses.append(Access(
                    node=n.target,
                    scope_id=self._scope_id_dict[scope],
                    definitions=defs
                ))

        self._defs_and_accesses = definitions, accesses
        return definitions, accesses


_metadata_cache: MutableMapping[AstNode, _AstMetadata] = weakref.WeakKeyDictionary()


def _get_metadata(ast_root: AstNode) -> _AstMetadata:
    """
    Gets (possibly cached) meta-data for the AST root provided.
    Stable behavior guaranteed only when ast_root is a module.
    :param ast_root:
    :return:
    """
    if ast_root in _metadata_cache:
        return _metadata_cache[ast_root]

    m = _metadata_cache[ast_root] = _AstMetadata.build(ast_root)
    return m


def get_uid_for_node(node: AstNode, context: AstNode) -> int:
    try:
        return _get_metadata(context).get_uid_for_node(node)
    except KeyError:
        raise KeyError("No UID associated with node in the given context.")


def get_node_for_uid(uid: int, context: AstNode) -> AstNode:
    try:
        return _get_metadata(context).get_node_for_uid(uid)
    except KeyError:
        raise KeyError("No node associated with UID in the given context.")


def get_parent(node: AstNode, context: AstNode) -> Optional[AstNode]:
    return _get_metadata(context).get_parent(node)


def iter_parents(node: AstNode, context: AstNode) -> Iterator[AstNode]:
    yield from _get_metadata(context).iter_parents(node)


def expr_has_load_ctx(expr_node: AstNode, context: AstNode) -> bool:
    return _get_metadata(context).get_expr_context(expr_node) == ExpressionContext.LOAD


def expr_has_store_ctx(expr_node: AstNode, context: AstNode) -> bool:
    return _get_metadata(context).get_expr_context(expr_node) == ExpressionContext.STORE


def expr_has_del_ctx(expr_node: AstNode, context: AstNode) -> bool:
    return _get_metadata(context).get_expr_context(expr_node) == ExpressionContext.DEL


def expr_is_evaluated(expr_node: AstNode, context: AstNode) -> bool:
    """
    Check whether the expression will actually evaluate to something during runtime.
    This is not the case for names/attributes/lookups that are targets of assignments,
    or name expressions that are used for argument names etc.
    :param expr_node:
    :param context:
    :return:
    """
    ctx = _get_metadata(context).get_expr_context(expr_node)
    return (ctx is None and not isinstance(expr_node, ExprContextEligible)) or ctx == ExpressionContext.LOAD


def iter_store_exprs(node: Union[AstStatementT, BaseExpression]) -> Iterator[BaseExpression]:
    if isinstance(node, AstStatement):
        module = cst.Module(body=[cst.SimpleStatementLine(body=[node])])
    else:
        module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])])

    wrapper = cst.metadata.MetadataWrapper(module,
                                           unsafe_skip_copy=True)

    expr_context = wrapper.resolve(ExpressionContextProvider)
    for n in walk(node):
        if expr_context.get(n, None) == ExpressionContext.STORE:
            yield n


def get_config_for_parsing(context: AstNode):
    return _get_metadata(context).get_config_for_parsing()


def get_node_uid_mapping(context: AstNode) -> Mapping[Union[AstNode, int], Union[int, AstNode]]:
    return _get_metadata(context).get_node_uid_mapping()


def get_scope_id_mapping(context: AstNode) -> Mapping[AstNode, int]:
    return _get_metadata(context).get_scope_id_mapping()


def get_definitions_and_accesses(ast_root: AstNode) -> typing.Tuple[typing.List[Definition], typing.List[Access]]:
    return _get_metadata(ast_root).get_definitions_and_accesses()


def parse(source_code: Union[str, dict], extension: str = '.py', python_version: str = None) -> Module:
    """
    Parses source code into an AST.
    The extension should be one of `.py` and `.ipynb`.
    The `python_version` argument should be one of '3.6', '3.7' and '3.8'.
    :param source_code:
    :param extension:
    :param python_version:
    :return:
    """
    if extension == '.ipynb':
        return parse_ipynb(source_code, python_version=python_version)

    if python_version is not None:
        config = cst.PartialParserConfig(encoding="utf-8", python_version=python_version)
    else:
        config = cst.PartialParserConfig(encoding="utf-8")

    return cst.parse_module(source_code, config=config)


def parse_stmt(source_code: str, config: cst.PartialParserConfig = None) -> AstStatementT:
    """
    Parse individual statements.
    :param source_code:
    :param config:
    :return:
    """
    if config is not None:
        result = cst.parse_statement(source_code, config=config)
    else:
        result = cst.parse_statement(source_code)

    if isinstance(result, cst.SimpleStatementLine):
        return result.body[0]

    return result


def parse_expr(source_code: str, config: cst.PartialParserConfig = None) -> cst.BaseExpression:
    """
    Parse individual expressions.
    :param source_code:
    :param config:
    :return:
    """
    if config is not None:
        return cst.parse_expression(source_code, config=config)

    return cst.parse_expression(source_code)


def to_code(ast_root: AstNode) -> str:
    """
    Returns source code corresponding to an AST node.
    :param ast_root:
    :return:
    """
    if isinstance(ast_root, cst.Module):
        return ast_root.code.strip()
    if isinstance(ast_root, AstNode):
        return cst.Module([]).code_for_node(ast_root).strip()

    raise TypeError(f"Cannot return code for node of type {type(ast_root)}")


def with_changes(node: AstNode, **changes):
    """
    Create a new node that is exactly like ``node'' but with certain attributes changed (as specified by `changes`)
    :param node:
    :param changes:
    :return:
    """
    return node.with_changes(**changes)


def with_deep_replacements(node: AstNode, replacements: typing.Dict[AstNode, AstNode]):
    #  If a node does not have a child (direct or indirect) in replacements, it should not be
    #  replaced for no reason. Hence add an identity mapping to replacements.
    replacements = replacements.copy()

    #  Replace Nones by empty sentinel
    replacements = {k: (v if v is not None else cst.RemoveFromParent()) for k, v in replacements.items()}

    par_mapping = collections.defaultdict(list)
    all_nodes = set(walk(node))
    for n in all_nodes:
        for c in n.children:
            par_mapping[c].append(n)

    worklist = collections.deque(replacements.keys())
    affected = set()
    while len(worklist) > 0:
        cur = worklist.popleft()
        if cur in affected:
            continue

        affected.add(cur)
        for par in par_mapping[cur]:
            if par not in affected:
                worklist.append(par)

    unaffected = all_nodes - affected
    replacements.update({n: n for n in unaffected})

    node = node.visit(ChildReplacementTransformer(replacements))
    return node


def wrap_with_parentheses(expr: BaseExpression):
    return expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _try_eval(node: AstNode) -> typing.Tuple[bool, Optional[typing.Any]]:
    try:
        val = eval(to_code(node), {}, {})
        return True, val
    except:
        return False, None


def is_constant(node: AstNode):
    if isinstance(node, (cst.BaseNumber, cst.BaseString)) or \
            (isinstance(node, Name) and (node.value == 'True' or node.value == 'False')):
        return True

    if isinstance(node, (cst.BaseList, cst.Tuple, cst.BaseDict, cst.BaseSet)):
        is_const, _ = _try_eval(node)
        return is_const

    return False


def get_constant_value(node: AstNode):
    if not is_constant(node):
        raise ValueError(f"Cannot evaluate node")

    return _try_eval(node)[1]


def is_expr(node: AstNode):
    return isinstance(node, cst.BaseExpression)


def is_stmt(node: AstNode):
    return isinstance(node, (cst.BaseSmallStatement, cst.BaseCompoundStatement))


def is_stmt_container(node: AstNode) -> bool:
    return isinstance(node, (cst.IndentedBlock, cst.SimpleStatementSuite, NotebookCellBody))


def iter_children(node: AstNode) -> Iterator[AstNode]:
    for c in node.children:
        yield c
        yield from iter_children(c)


def iter_stmts(node: AstNode) -> Iterator[AstStatementT]:
    for n in walk(node):
        if is_stmt(n):
            yield n


def iter_true_exprs(node: AstNode, context: AstNode) -> Iterator[BaseExpression]:
    for n in walk(node):
        if isinstance(n, BaseExpression) and expr_is_evaluated(n, context):
            yield n


def get_assign_targets(node: Union[Assign, AugAssign, AnnAssign]) -> typing.List[BaseExpression]:
    module = cst.Module(body=[cst.SimpleStatementLine(body=[node])])
    wrapper = cst.metadata.MetadataWrapper(module, unsafe_skip_copy=True)

    expr_context = wrapper.resolve(ExpressionContextProvider)
    if isinstance(node, Assign):
        node_iter = itertools.chain(*(walk(t.target) for t in node.targets))
    else:
        node_iter = walk(node.target)
    return [n for n in node_iter
            if expr_context.get(n, None) == ExpressionContext.STORE and isinstance(n, (Name, Subscript, Attribute))]


def iter_body_stmts(node: AstNode) -> Iterator[AstStatementT]:
    if hasattr(node, "body") and isinstance(node.body, cst.BaseSuite):
        node = node.body

    assert isinstance(node, Module) or is_stmt_container(node)

    for stmt in node.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            yield from stmt.body
        else:
            yield stmt


def prepare_body(stmts: typing.List[AstStatementT]) -> typing.List[Union[cst.SimpleStatementLine,
                                                                         cst.BaseCompoundStatement]]:
    """
    Pre-processing to ensure stmts can become part of a body of a stmt container.
    :param stmts:
    :return:
    """
    body = []
    for s in stmts:
        if isinstance(s, cst.BaseCompoundStatement):
            body.append(s)
        elif isinstance(s, cst.BaseSmallStatement):
            body.append(cst.SimpleStatementLine(body=[s]))
        elif isinstance(s, cst.BaseExpression):
            body.append(cst.SimpleStatementLine(body=[cst.Expr(value=s)]))
        else:
            raise TypeError(f"Cannot include statement of type {type(s)}.")

    return body


def update_stmt_body(node: AstNode, new_body: typing.List[cst.BaseStatement]) -> AstNode:
    if isinstance(node, (Module, cst.IndentedBlock, NotebookCellBody)):
        return node.with_changes(body=new_body)

    elif isinstance(node, cst.SimpleStatementSuite):
        return cst.IndentedBlock(body=new_body)

    elif hasattr(node, "body") and isinstance(node.body, cst.BaseSuite):
        return node.with_changes(body=update_stmt_body(node.body, new_body))

    else:
        raise TypeError(f"Cannot update body of node with type {type(node)}")


def is_starred_expr(expr_node: AstNode) -> bool:
    return isinstance(expr_node, (cst.StarredElement, cst.StarredDictElement))


def create_name_expr(name: str) -> Name:
    return Name(value=name)


def create_subscript_expr(value: Union[str, BaseExpression], slices: typing.List[Union[str, BaseExpression]]):
    if isinstance(value, str):
        value = parse_expr(value)

    slices = [parse_expr(s) if isinstance(s, str) else s for s in slices]
    slice_nodes = [cst.SubscriptElement(slice=cst.Index(s)) for s in slices]
    return cst.Subscript(value=value, slice=slice_nodes)


def wrap_try_finally(body, finalbody):
    return cst.Try(body=cst.IndentedBlock(body=body),
                   finalbody=cst.Finally(body=cst.IndentedBlock(body=finalbody)))


def create_list_expr(element_exprs: typing.List[BaseExpression]):
    return cst.List(elements=[cst.Element(value=e) for e in element_exprs])


def create_tuple_expr(element_exprs: typing.List[BaseExpression]):
    return cst.Tuple(elements=[cst.Element(value=e) for e in element_exprs])


def create_assignment(targets: typing.List[Union[str, BaseAssignTargetExpression]],
                      value: Union[str, AstNode]) -> Assign:
    targets = [t if isinstance(t, BaseAssignTargetExpression) else parse_expr(t) for t in targets]
    if isinstance(value, str):
        value = parse_expr(value)

    return Assign(targets=[cst.AssignTarget(target=t) for t in targets],
                  value=value)


def wrap_with_call(node: Union[str, BaseExpression], func: Union[str, BaseExpression]) -> Call:
    if isinstance(node, str):
        node = parse_expr(node)

    if isinstance(func, str):
        func = parse_expr(func)

    return cst.Call(func=func, args=[cst.Arg(value=node, keyword=None, star='')])


def create_return(value: BaseExpression) -> Return:
    return cst.Return(value=value)


def remove_simple_statements_and_simplify(node: AstNode,
                                          to_remove: Optional[typing.Set[BaseSmallStatement]] = None,
                                          to_retain: Optional[typing.Set[BaseSmallStatement]] = None):
    if (to_remove is None and to_retain is None) or (to_remove is not None and to_retain is not None):
        raise AssertionError("Exactly one of to_remove and to_retain has to be provided.")

    return node.visit(StmtRemovalAndSimplificationTransformer(to_remove=to_remove, to_retain=to_retain))


def remove_nodes_from_ast(ast_: AstNode, to_remove: typing.Collection[AstNode]) -> AstNode:
    return ast_.visit(NodeRemovalTransformer(to_remove))
