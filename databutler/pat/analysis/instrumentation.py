"""
A Simple Instrumentation Framework
"""
import collections
import inspect
import itertools
import textwrap
from abc import ABC

import attr
from typing import List, Callable, Any, Dict, Iterator, Optional, Tuple, Union, Set

from databutler.pat import astlib
from databutler.pat.utils import codeutils
from databutler.utils.logging import logger


@attr.s(cmp=False, repr=False)
class BaseGenerator(ABC):
    def preprocess(self, ast_root: astlib.AstNode):
        """
        A subclass can override this method to setup/reset necessary data-structures
        before callbacks/decorators/wrappers are generated using the respective routines.
        This is guaranteed to be called, before the crucial methods such as
        gen_callbacks/gen_wrappers etc. are invoked.
        :param ast_root:
        :return:
        """


@attr.s(cmp=False, repr=False)
class StmtCallback:
    """
    Callbacks for AST statements i.e. nodes subclassing cst.BaseStatement.
    Callbacks can be invoked before ('pre') or after ('post') a statement.
    Callbacks receive the values of globals() and locals() respectively as positional arguments.
    Name must be unique to every callback within an AST.

    The `mandatory` attribute indicates whether, for post-callbacks, the callable
    should execute even if the control flow changes right after the statement is executed.
    This could happen due to the statement being a
    (return/break/continue) statement, or the statement raising an exception.
    The instrumenter handles mandatory callbacks using a try-finally block, and hence this is False by
    default to avoid too many try-finally blocks.
    """
    callable: Callable[[Dict, Dict], Any] = attr.ib()
    name: str = attr.ib()

    position: str = attr.ib(default='pre')  # 'pre' or 'post'
    mandatory: bool = attr.ib(default=False)  # Ignored if position is 'pre'
    # If not default, responsibility of the user to make sure there are no errors
    arg_str: str = attr.ib(default="globals(), locals()")

    #  Priorities are used to enforce an ordering amongst statement callbacks, even if instrumentation
    #  passes are in a different order. Use sparingly.
    priority: int = attr.ib(default=0)


@attr.s(cmp=False, repr=False)
class StmtCallbacksGenerator(BaseGenerator):
    """
    Base Generator for statement callbacks for a selection of AST nodes (of type ast.stmt).
    """
    #  Internal Stuff
    _callback_id: int = attr.ib(init=False, default=0)

    def gen_stmt_callback_id(self):
        """
        Simple utility to generate callback names if one does not want to write their own.
        :return:
        """
        self._callback_id += 1
        return f"__automl_stmt_callback_{self.__class__.__name__}_{self._callback_id:05}___"

    def iter_stmts(self, ast_root: astlib.AstNode) -> Iterator[astlib.AstStatementT]:
        """
        Useful utility to iterate over statements.
        :param ast_root:
        :return:
        """
        for n in astlib.walk(ast_root):
            if astlib.is_stmt(n):
                yield n

    def gen_stmt_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.AstStatementT,
                                                                   List[StmtCallback]]:
        return {}


@attr.s(cmp=False, repr=False)
class ExprCallback:
    """
    Callbacks for AST expressions. Like statement callbacks, can be pre / post.
    Pre-callbacks execute before the expression is evaluated. This includes sub-expressions i.e.
    the callback will be executed before any of the sub-expressions are evaluated as well.

    Callbacks receive the values of globals() and locals() respectively as positional arguments.
    Name must be unique to every callback within an AST.
    """
    callable: Callable[[Dict, Dict], Any] = attr.ib()
    name: str = attr.ib()

    position: str = attr.ib(default='pre')  # 'pre' or 'post'
    # If not default, responsibility of the user to make sure there are no errors
    arg_str: str = attr.ib(default="globals(), locals()")

    #  Priorities are used to enforce an ordering amongst statement callbacks, even if instrumentation
    #  passes are in a different order. Use sparingly.
    priority: int = attr.ib(default=0)


@attr.s(cmp=False, repr=False)
class ExprCallbacksGenerator(BaseGenerator):
    """
    Base Generator for expression callbacks for a selection of AST nodes (of type ast.expr).
    """
    #  Internal Stuff
    _callback_id: int = attr.ib(init=False, default=0)

    def gen_expr_callback_id(self):
        """
        Simple utility to generate wrapper names if one does not want to write their own.
        :return:
        """
        self._callback_id += 1
        return f"__automl_expr_callback{self.__class__.__name__}_{self._callback_id:05}___"

    def iter_valid_exprs(self,
                         ast_root: astlib.AstNode,
                         context: Optional[astlib.AstNode] = None) -> Iterator[astlib.BaseExpression]:
        if context is None:
            context = ast_root

        for n in astlib.walk(ast_root):
            if isinstance(n, astlib.BaseExpression):
                if astlib.expr_is_evaluated(n, context=context):
                    yield n

    def gen_expr_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        return {}


@attr.s(cmp=False, repr=False)
class ExprWrapper:
    """
    Wrappers for expressions i.e. AST nodes subclassing ast.expr.
    The first argument to the wrapper must be the value of the wrapped expression,
    which they return as is, or modify accordingly.
    The rest of the arguments will be passed according to the `arg_str` parameter.
    Name must be unique to every wrapper within an AST.
    """
    callable: Callable[[Any], Any] = attr.ib()
    name: str = attr.ib()
    arg_str: str = attr.ib(default='')


@attr.s(cmp=False, repr=False)
class ExprWrappersGenerator(BaseGenerator):
    """
    Base Generator for expression wrappers for a selection of AST nodes (of type ast.expr).
    """
    #  Internal Stuff
    _wrapper_id: int = attr.ib(init=False, default=0)

    def gen_wrapper_id(self):
        """
        Simple utility to generate wrapper names if one does not want to write their own.
        :return:
        """
        self._wrapper_id += 1
        return f"__automl_expr_wrapper_{self.__class__.__name__}_{self._wrapper_id:05}___"

    def iter_valid_exprs(self,
                         ast_root: astlib.AstNode,
                         context: Optional[astlib.AstNode] = None) -> Iterator[astlib.BaseExpression]:
        if context is None:
            context = ast_root

        for n in astlib.walk(ast_root):
            if isinstance(n, astlib.BaseExpression):
                if astlib.expr_is_evaluated(n, context=context):
                    yield n

    def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        return {}


@attr.s(cmp=False, repr=False)
class CallDecorator:
    """
    A decorator for a function. Should have a callable that takes in (func, args, kwargs) as arguments, and
    either do not return anything, or return the desired result of the overall call.
    The returned result may be something different, and hence this can be used for interception.
    If it does return nothing, the does_not_return attribute should be set to True (which it is by default).
    One can also return new args and kwargs to pass to later decorators or the final function.
    This can be enabled by setting `returns_new_args` to True.
    One can access the return value by setting `needs_return_value` to True. In this case the arguments to the
    callable will be (func, ret_val, args, kwargs). Cannot be used simultaneously with
    `returns_new_args=True` or `does_not_return=False`
    """
    callable: Union[Callable[[Callable, Tuple[Any], Dict[str, Any]], Any],
                    Callable[[Callable, Any, Tuple[Any], Dict[str, Any]], Any]] = attr.ib()
    does_not_return: bool = attr.ib(default=True)
    returns_new_args: bool = attr.ib(default=False)
    needs_return_value: bool = attr.ib(default=False)


@attr.s(cmp=False, repr=False)
class CallDecoratorsGenerator(BaseGenerator):
    """
    Base Generator for decorators for a selection of AST nodes (of type ast.Call).
    """

    def iter_funcs(self, ast_root: astlib.AstNode) -> Iterator[astlib.BaseExpression]:
        for n in astlib.walk(ast_root):
            if isinstance(n, astlib.Call):
                yield n.func

    def iter_calls(self, ast_root: astlib.AstNode) -> Iterator[astlib.Call]:
        for n in astlib.walk(ast_root):
            if isinstance(n, astlib.Call):
                yield n

    def gen_decorators(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        """
        Note that the decorator will be applied to the evaluation of the n.func attribute when n is of
        type ast.Call, and the mapping returned from this function should have the key
        as n.func, not the ast.Call node.
        :param ast_root:
        :return:
        """
        return {}

    @staticmethod
    def gen_master_decorator(decorators: List[CallDecorator]) -> Callable:
        non_returning_decorators: List[CallDecorator] = []
        returning_decorator: Optional[CallDecorator] = None
        retval_needing_decorators: List[CallDecorator] = []

        #  Only the first returning decorator is kept.
        #  Make all the non-returning decorators execute first, while maintaining the order.
        #  Note that this means returning decorators will be executed last, no matter what.
        for d in decorators:
            if d.does_not_return:
                if d.needs_return_value:
                    retval_needing_decorators.append(d)
                else:
                    non_returning_decorators.append(d)

            elif returning_decorator is None:
                returning_decorator = d

        def master_wrapper(func: Callable):
            def wrapper(*args, **kwargs):
                for d in non_returning_decorators:
                    if d.returns_new_args:
                        args, kwargs = d.callable(func, args, kwargs)
                    else:
                        d.callable(func, args, kwargs)

                if returning_decorator is not None:
                    result = returning_decorator.callable(func, args, kwargs)
                else:
                    result = func(*args, **kwargs)

                for d in retval_needing_decorators:
                    d.callable(func, result, args, kwargs)

                return result

            return wrapper

        return master_wrapper


@attr.s(cmp=False, repr=False)
class Instrumentation:
    stmt_callback_gens: List[StmtCallbacksGenerator] = attr.ib(factory=list)
    expr_callback_gens: List[ExprCallbacksGenerator] = attr.ib(factory=list)
    expr_wrapper_gens: List[ExprWrappersGenerator] = attr.ib(factory=list)
    call_decorator_gens: List[CallDecoratorsGenerator] = attr.ib(factory=list)

    def preprocess(self, ast_root: astlib.AstNode):
        """
        Help base generators setup/reset data-structures before they are asked
        to generate callbacks/decorators/wrappers.
        :param ast_root:
        :return:
        """
        for base_gen in set(itertools.chain(self.stmt_callback_gens,
                                            self.expr_callback_gens,
                                            self.expr_wrapper_gens,
                                            self.call_decorator_gens)):
            base_gen.preprocess(ast_root)

    def __or__(self, other):
        if not isinstance(other, Instrumentation):
            raise TypeError(f"Can only '|' between two objects of type {self.__class__.__name__}")

        return Instrumentation(
            stmt_callback_gens=self.stmt_callback_gens + other.stmt_callback_gens,
            expr_callback_gens=self.expr_callback_gens + other.expr_callback_gens,
            expr_wrapper_gens=self.expr_wrapper_gens + other.expr_wrapper_gens,
            call_decorator_gens=self.call_decorator_gens + other.call_decorator_gens,
        )

    @classmethod
    def from_generators(cls, *generators):
        stmt_callback_gens = [g for g in generators if isinstance(g, StmtCallbacksGenerator)]
        expr_callback_gens = [g for g in generators if isinstance(g, ExprCallbacksGenerator)]
        expr_wrapper_gens = [g for g in generators if isinstance(g, ExprWrappersGenerator)]
        call_decorator_gens = [g for g in generators if isinstance(g, CallDecoratorsGenerator)]

        return cls(
            stmt_callback_gens=stmt_callback_gens,
            expr_wrapper_gens=expr_wrapper_gens,
            expr_callback_gens=expr_callback_gens,
            call_decorator_gens=call_decorator_gens,
        )


@attr.s(cmp=False, repr=False)
class Instrumenter(astlib.AstTransformer):
    """
    The actual instrumentation engine.
    Takes as arguments a list of statement-callback generators and expression-wrapper generators.
    Can be extended to other types of AST nodes if need be in the future.
    """
    instrumentation: Instrumentation = attr.ib(factory=Instrumentation)

    #  Internal
    _stmt_callbacks: Dict[astlib.AstStatementT, List[StmtCallback]] = attr.ib(init=False, default=None)
    _expr_callbacks: Dict[astlib.BaseExpression, List[ExprCallback]] = attr.ib(init=False, default=None)
    _expr_wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = attr.ib(init=False, default=None)
    _call_decorators: Dict[astlib.BaseExpression, List[CallDecorator]] = attr.ib(init=False, default=None)
    _callable_mapping: Dict[str, Callable] = attr.ib(init=False, factory=dict)
    _master_decorator_id: int = attr.ib(init=False, default=0)
    _master_wrapper_id: int = attr.ib(init=False, default=0)
    _cur_config = attr.ib(init=False, default=None)
    _node_mapping = attr.ib(init=False, default=None)
    _uninstrumentable_exprs: Set[astlib.BaseExpression] = attr.ib(init=False, default=None)
    _decorator_exprs: Set[astlib.BaseExpression] = attr.ib(init=False, default=None)
    _cur_ast: astlib.AstNode = attr.ib(init=False, default=None)

    def _gen_master_decorator_id(self):
        """
        Simple utility to generate names for master decorators.
        :return:
        """
        self._master_decorator_id += 1
        return f"__automl_master_decorator_{self.__class__.__name__}_{self._master_decorator_id:05}___"

    def _gen_master_wrapper_id(self):
        """
        Simple utility to generate names for master wrappers.
        :return:
        """
        self._master_wrapper_id += 1
        return f"__automl_master_wrapper_{self.__class__.__name__}_{self._master_wrapper_id:05}___"

    def _prepare_instrumentation(self, ast_root: astlib.AstNode):
        self._stmt_callbacks = collections.defaultdict(list)
        self._expr_callbacks = collections.defaultdict(list)
        self._expr_wrappers = collections.defaultdict(list)
        self._call_decorators = collections.defaultdict(list)
        inst = self.instrumentation
        inst.preprocess(ast_root)

        for stmt_callback_mapping in (c_gen.gen_stmt_callbacks(ast_root) for c_gen in inst.stmt_callback_gens):
            for k, v in stmt_callback_mapping.items():
                self._stmt_callbacks[k].extend(v)

        for expr_callback_mapping in (c_gen.gen_expr_callbacks(ast_root) for c_gen in inst.expr_callback_gens):
            for k, v in expr_callback_mapping.items():
                self._expr_callbacks[k].extend(v)

        for expr_wrapper_mapping in (w_gen.gen_expr_wrappers(ast_root) for w_gen in inst.expr_wrapper_gens):
            for k, v in expr_wrapper_mapping.items():
                self._expr_wrappers[k].extend(v)

        for call_decorator_mapping in (w_gen.gen_decorators(ast_root) for w_gen in inst.call_decorator_gens):
            for k, v in call_decorator_mapping.items():
                self._call_decorators[k].extend(v)

        self._cur_config = astlib.get_config_for_parsing(ast_root)
        self._node_mapping = {}
        self._uninstrumentable_exprs = set()
        self._decorator_exprs = set()
        self._cur_ast = ast_root

    def _add_instrumentation(self, ast_root: astlib.AstNode) -> astlib.Module:
        """
        This assumes `prepare_instrumentation` has been called.
        :param ast_root:
        :return:
        """
        return ast_root.visit(self)

    def process(self, ast_root: astlib.AstNode) -> Tuple[astlib.Module, Dict[str, Callable]]:
        """
        Please call this instead of calling `visit` directly.
        :param ast_root:
        :return:
        """
        self._prepare_instrumentation(ast_root)
        return self._add_instrumentation(ast_root), self._callable_mapping.copy()

    def process_func(self, func: Callable, as_module: bool = False) -> Tuple[astlib.Module, Callable]:
        """
        Instrument an existing function. Returns the (un-transformed) AST corresponding to the function,
        and the new instrumented version of the function that can be used directly.
        This procedure takes care of handling closures as well as global mappings resulting from the instrumntation.
        Logic for closures borrowed from the atlas project (https://github.com/rbavishi/atlas)

        NOTE : `inspect.getsource` needs to execute successfully on the function.
        :param func:
        :param as_module:
        :return:
        """
        try:
            source_code = textwrap.dedent(inspect.getsource(func))

        except Exception as e:
            logger.error("Could not run inspect.getsource on passed function.")
            logger.exception(e)
            raise

        ast_root: astlib.Module = astlib.parse(source_code)
        self._prepare_instrumentation(ast_root)
        #  Only keep instrumentation within the function.
        if not as_module:
            for s in ast_root.body:
                if s in self._stmt_callbacks:
                    self._stmt_callbacks.pop(s)

        ast_root = self._add_instrumentation(ast_root)
        callable_mapping = self._callable_mapping.copy()

        #  Get all the external dependencies of this function.
        #  We rely on a modified closure function adopted from the ``inspect`` library.
        closure_vars = codeutils.getclosurevars_recursive(func)
        globs: Dict[str, Any] = {**closure_vars.nonlocals.copy(), **closure_vars.globals.copy()}
        globs.update(callable_mapping)

        filename = inspect.getabsfile(func)
        exec(compile(astlib.to_code(ast_root), filename=filename, mode="exec"), globs)
        result = globs[func.__name__]
        globs["__name__"] = filename

        if inspect.ismethod(func):
            result = result.__get__(func.__self__, func.__self__.__class__)

        return ast_root, result

    def on_visit(self, node: astlib.AstNode):
        if isinstance(node, astlib.ConcatenatedString):
            self._uninstrumentable_exprs.add(node.left)
            self._uninstrumentable_exprs.add(node.right)

        if astlib.is_starred_expr(node):
            self._uninstrumentable_exprs.add(node)

        if isinstance(node, astlib.Decorator):
            self._decorator_exprs.add(node.decorator)

        return True

    def on_leave(self,
                 original_node: astlib.AstNode,
                 updated_node: astlib.AstNode):

        if isinstance(original_node, astlib.Module) or astlib.is_stmt_container(original_node):
            assert isinstance(updated_node, astlib.Module) or astlib.is_stmt_container(updated_node)
            new_body = []

            for stmt in astlib.iter_body_stmts(updated_node):
                new_body.extend(self._process_stmt(stmt))

            return astlib.update_stmt_body(updated_node, new_body)

        elif astlib.is_stmt(original_node):
            self._node_mapping[updated_node] = original_node
            return updated_node

        elif isinstance(original_node, astlib.BaseExpression):
            assert isinstance(updated_node, astlib.BaseExpression)
            if astlib.expr_is_evaluated(original_node, context=self._cur_ast):
                return self._process_expr(original_node, updated_node)
            else:
                callbacks = self._expr_callbacks.get(original_node, [])
                wrappers = self._expr_wrappers.get(original_node, [])
                if len(callbacks) > 0:
                    logger.warning("Expression callbacks specified for expression with invalid ctx.")
                if len(wrappers) > 0:
                    logger.warning("Expression wrappers specified for expression with invalid ctx.")

        return updated_node

    def _process_stmt(self, stmt: astlib.AstStatementT):
        key = self._node_mapping[stmt]

        stmt_callbacks = sorted(self._stmt_callbacks.get(key, []), key=lambda x: -x.priority)
        pre_callbacks = [c for c in stmt_callbacks if c.position == 'pre']
        post_callbacks = [c for c in stmt_callbacks if c.position == 'post' and not c.mandatory]
        mandatory_post_callbacks = [c for c in stmt_callbacks if c.position == 'post' and c.mandatory]

        node_sequence: List[Union[astlib.AstStatementT]] = []
        for cb in pre_callbacks:
            self._callable_mapping[cb.name] = cb.callable
            node_sequence.append(astlib.parse_stmt(f"{cb.name}({cb.arg_str})",
                                                   config=self._cur_config))

        node_sequence.append(stmt)

        for cb in post_callbacks:
            self._callable_mapping[cb.name] = cb.callable
            node_sequence.append(astlib.parse_stmt(f"{cb.name}({cb.arg_str})",
                                                   config=self._cur_config))

        node_sequence = astlib.prepare_body(node_sequence)

        if len(mandatory_post_callbacks) > 0:
            #  Wrap the statements in a try-finally block.
            final_sequence = []
            for cb in mandatory_post_callbacks:
                self._callable_mapping[cb.name] = cb.callable
                final_sequence.append(astlib.parse_stmt(f"{cb.name}({cb.arg_str})",
                                                        config=self._cur_config))

            final_sequence = astlib.prepare_body(final_sequence)
            try_stmt = astlib.wrap_try_finally(body=node_sequence, finalbody=final_sequence)
            return astlib.prepare_body([try_stmt])

        return node_sequence

    def _process_expr(self,
                      orig_expr: astlib.BaseExpression,
                      updated_expr: astlib.BaseExpression) -> astlib.BaseExpression:
        if orig_expr in self._uninstrumentable_exprs:
            return updated_expr

        wrappers = self._expr_wrappers.get(orig_expr, [])
        value_callables_dict = collections.defaultdict(list)
        for w in wrappers:
            value_callables_dict[w.arg_str].append(w.callable)

        #  After wrappers, call any decorators if they exist.
        if orig_expr in self._call_decorators:
            master_decorator = CallDecoratorsGenerator.gen_master_decorator(self._call_decorators[orig_expr])
            value_callables_dict[''].append(master_decorator)

        result = updated_expr
        for arg_str, value_callables in value_callables_dict.items():
            wrapper_name = self._gen_master_wrapper_id()

            if len(value_callables) > 1:
                _master_callable = self._get_master_callable(value_callables)
                self._callable_mapping[wrapper_name] = _master_callable
                func = astlib.parse_expr(f"{wrapper_name}({arg_str})")
                result = func.with_changes(args=[astlib.Arg(value=astlib.wrap_with_parentheses(result)),
                                                 *func.args])

            elif len(value_callables) == 1:
                self._callable_mapping[wrapper_name] = value_callables[0]
                func = astlib.parse_expr(f"{wrapper_name}({arg_str})")
                result = func.with_changes(args=[astlib.Arg(value=astlib.wrap_with_parentheses(result)),
                                                 *func.args])

        #  Get callbacks for the expression
        callbacks = sorted(self._expr_callbacks.get(orig_expr, []), key=lambda x: -x.priority)
        pre_callbacks = [c for c in callbacks if c.position == 'pre']
        post_callbacks = [c for c in callbacks if c.position == 'post']

        if len(pre_callbacks) == 0 and len(post_callbacks) == 0:
            return result

        pre_sequence = []
        post_sequence = []

        for cb in pre_callbacks:
            self._callable_mapping[cb.name] = cb.callable
            pre_sequence.append(astlib.parse_expr(f"{cb.name}({cb.arg_str})",
                                                  config=self._cur_config))

        for cb in post_callbacks:
            self._callable_mapping[cb.name] = cb.callable
            post_sequence.append(astlib.parse_expr(f"{cb.name}({cb.arg_str})",
                                                   config=self._cur_config))

        if len(pre_sequence) == 0:
            post_expr = astlib.create_list_expr(element_exprs=post_sequence)
            tuple_expr = astlib.create_tuple_expr(element_exprs=[result, post_expr])
            index_expr = astlib.parse_expr("dummy[0]")
            new_node = astlib.with_changes(index_expr, value=tuple_expr)

        elif len(post_sequence) == 0:
            pre_expr = astlib.create_list_expr(element_exprs=pre_sequence)
            tuple_expr = astlib.create_tuple_expr(element_exprs=[pre_expr, result])
            index_expr = astlib.parse_expr("dummy[1]")
            new_node = astlib.with_changes(index_expr, value=tuple_expr)

        else:
            pre_expr = astlib.create_list_expr(element_exprs=pre_sequence)
            post_expr = astlib.create_list_expr(element_exprs=post_sequence)
            tuple_expr = astlib.create_tuple_expr(element_exprs=[pre_expr, result, post_expr])
            index_expr = astlib.parse_expr("dummy[1]")
            new_node = astlib.with_changes(index_expr, value=tuple_expr)

        if orig_expr in self._decorator_exprs:
            new_node = astlib.wrap_with_call(new_node, "__instrumentation_expr_passthrough")
            self._callable_mapping["__instrumentation_expr_passthrough"] = self._expr_passthrough

        return new_node

    @staticmethod
    def _expr_passthrough(value):
        return value

    def _get_master_callable(self, callables):
        def _master_callable(in_value, *args, **kwargs):
            for func in callables:
                in_value = func(in_value, *args, **kwargs)

            return in_value

        return _master_callable
