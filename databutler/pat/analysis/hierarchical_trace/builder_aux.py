import collections
import inspect
import itertools
from typing import List, Dict, Union, Any, Callable, Type, Tuple, Set

import attr

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder_utils import (
    TraceEventsCollector,
    TraceItemsCollector,
    TemporaryVariablesGenerator,
)
from databutler.pat.analysis.hierarchical_trace.core import (
    TraceItem,
    DefEvent,
    AccessEvent,
    ObjWriteEvent,
    ObjReadEvent,
    TraceItemType,
)
from databutler.pat.analysis.instrumentation import (
    StmtCallbacksGenerator,
    CallDecoratorsGenerator,
    ExprWrappersGenerator,
    ExprCallbacksGenerator,
    BaseGenerator,
    StmtCallback,
    ExprWrapper,
    ExprCallback,
    CallDecorator,
)
from databutler.pat.utils import miscutils
from databutler.utils.logging import logger


@attr.s(cmp=False, repr=False)
class FuncDefLambdaAuxGenerator(
    StmtCallbacksGenerator,
    CallDecoratorsGenerator,
    ExprWrappersGenerator,
    ExprCallbacksGenerator,
):
    """
    Auxiliary trace items plus events generator for functions and Lambdas
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()
    temp_vars_generator: TemporaryVariablesGenerator = attr.ib()

    #  Internal Stuff
    _definitions: List[astlib.Definition] = attr.ib(init=False, default=None)
    _scope_id_dict: Dict[astlib.AstNode, astlib.ScopeId] = attr.ib(
        init=False, default=None
    )
    _callsite_records: Dict[Union[astlib.FunctionDef, astlib.Lambda], Dict] = attr.ib(
        init=False, default=None
    )
    _last_known_expr_value: Dict[astlib.BaseExpression, Any] = attr.ib(
        init=False, default=None
    )
    _func_lambdas_encountered: Dict[
        Callable, Union[astlib.FunctionDef, astlib.Lambda]
    ] = attr.ib(init=False, default=None)
    _aux_instrumentation: Dict[Type[BaseGenerator], Dict] = attr.ib(
        init=False, default=None
    )

    def __attrs_post_init__(self):
        self.reset()

    def reset(self):
        self._func_lambdas_encountered = {}
        self._callsite_records = {}
        self._last_known_expr_value = {}
        self._scope_id_dict = {}
        self._definitions = []

    def preprocess(self, ast_root: astlib.AstNode):
        self._scope_id_dict.update(astlib.get_scope_id_mapping(ast_root))
        self._definitions.extend(astlib.get_definitions_and_accesses(ast_root)[0])

        self._aux_instrumentation = {
            StmtCallbacksGenerator: collections.defaultdict(list),
            ExprWrappersGenerator: collections.defaultdict(list),
            ExprCallbacksGenerator: collections.defaultdict(list),
            CallDecoratorsGenerator: collections.defaultdict(list),
        }

        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                for k, v in self._gen_aux_instrumentation_call_expr(expr).items():
                    miscutils.merge_defaultdicts_list(self._aux_instrumentation[k], v)

            elif isinstance(expr, astlib.Lambda):
                for k, v in self._gen_aux_instrumentation_lambda(expr).items():
                    miscutils.merge_defaultdicts_list(self._aux_instrumentation[k], v)

        for stmt in self.iter_stmts(ast_root):
            if isinstance(stmt, astlib.FunctionDef):
                for k, v in self._gen_aux_instrumentation_func_def(stmt).items():
                    miscutils.merge_defaultdicts_list(self._aux_instrumentation[k], v)

    def gen_stmt_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
        callbacks: Dict[
            astlib.AstStatementT, List[StmtCallback]
        ] = collections.defaultdict(list)

        for stmt in self.iter_stmts(ast_root):
            if isinstance(stmt, astlib.FunctionDef):
                callbacks[stmt].append(self._gen_func_def_tracker(stmt))

        miscutils.merge_defaultdicts_list(
            callbacks, self._aux_instrumentation[StmtCallbacksGenerator]
        )
        return callbacks

    def gen_expr_wrappers(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers: Dict[
            astlib.BaseExpression, List[ExprWrapper]
        ] = collections.defaultdict(list)

        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Lambda):
                wrappers[expr].append(self._gen_lambda_def_tracker(expr))

        miscutils.merge_defaultdicts_list(
            wrappers, self._aux_instrumentation[ExprWrappersGenerator]
        )
        return wrappers

    def gen_expr_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        callbacks: Dict[
            astlib.BaseExpression, List[ExprCallback]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            callbacks, self._aux_instrumentation[ExprCallbacksGenerator]
        )
        return callbacks

    def gen_decorators(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[
            astlib.BaseExpression, List[CallDecorator]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            decorators, self._aux_instrumentation[CallDecoratorsGenerator]
        )
        return decorators

    def _gen_func_def_tracker(self, f_def: astlib.FunctionDef) -> StmtCallback:
        name = f_def.name.value

        def callback(d_globals, d_locals):
            func = d_locals.get(name, d_globals.get(name, None))
            if func is not None:
                self._func_lambdas_encountered[func] = f_def

        return StmtCallback(
            callable=callback,
            name=self.gen_stmt_callback_id(),
            position="post",
            mandatory=False,
            arg_str="globals(), locals()",
        )

    def _gen_lambda_def_tracker(self, expr: astlib.Lambda) -> ExprWrapper:
        def wrapper(func):
            self._func_lambdas_encountered[func] = expr
            return func

        return ExprWrapper(callable=wrapper, name=self.gen_wrapper_id())

    def _gen_param_assignment(
        self,
        name: str,
        obj: Any,
        accesses: Set[str],
        aux_node: astlib.AstNode,
        scope_id: int,
        definition: astlib.Definition,
    ):
        start_time = self.clock.get_time()
        access_time = self.clock.increment(step=1)
        def_time = self.clock.increment(step=1)
        end_time = self.clock.increment(step=1)
        trace_item = TraceItem(
            start_time=start_time,
            end_time=end_time,
            ast_node=aux_node,
            item_type=TraceItemType.AUX,
            scope_id=scope_id,
            metadata={"aux-type": "param-assignment"},
        )
        self.trace_items_collector.add_item(trace_item)

        for a_name in accesses:
            def_event = self.trace_events_collector.get_last_event_for_def(a_name)
            if def_event is None:
                logger.warning(f"Could not find def event for {a_name}")
                continue

            event = AccessEvent(
                timestamp=access_time,
                owner=trace_item,
                obj_id=def_event.obj_id,
                name=a_name,
                scope_id=scope_id,
                def_event=def_event,
                ast_node=trace_item.ast_node,
            )
            self.trace_events_collector.add_event(event)

        def_event = DefEvent(
            timestamp=def_time,
            owner=trace_item,
            obj_id=id(obj),
            name=name,
            scope_id=scope_id,
            def_key=definition,
            ast_node=trace_item.ast_node,
        )
        self.trace_events_collector.add_event(def_event, obj=obj)

    def _gen_param_assignment_regular(
        self,
        name: str,
        obj: Any,
        sym_value,
        scope_id: int,
        definition: astlib.Definition,
    ):
        if isinstance(sym_value, str):
            aux_node = astlib.create_assignment([name], sym_value)
            accesses = {sym_value}
        elif isinstance(sym_value, tuple):
            aux_node = astlib.create_assignment(
                [name], f"{sym_value[0]}[{sym_value[1]!r}]"
            )
            accesses = {sym_value[0]}
        else:
            raise AssertionError

        self._gen_param_assignment(name, obj, accesses, aux_node, scope_id, definition)

    def _gen_param_assignment_var(
        self,
        name: str,
        obj: Any,
        sym_value: Tuple,
        scope_id: int,
        definition: astlib.Definition,
    ):
        if isinstance(sym_value, tuple):
            elems = [
                s if isinstance(s, str) else f"{s[0]}[{s[1]!r}]" for s in sym_value
            ]
            accesses = {s if isinstance(s, str) else s[0] for s in sym_value}
            value = "(" + ", ".join(elems) + ",)"
        elif isinstance(sym_value, dict):
            elems = {
                k: (s if isinstance(s, str) else f"{s[0]}[{s[1]!r}]")
                for k, s in sym_value.items()
            }
            accesses = {s if isinstance(s, str) else s[0] for s in sym_value.values()}
            value = "{" + ", ".join(f"{k!r}: {v}" for k, v in elems.items()) + "}"
        else:
            raise AssertionError

        aux_node = astlib.create_assignment([name], value)
        self._gen_param_assignment(name, obj, accesses, aux_node, scope_id, definition)

    def _gen_param_assignment_defaults(
        self, name: str, obj: Any, scope_id: int, definition: astlib.Definition
    ):
        aux_node = astlib.create_assignment([name], repr(obj))
        self._gen_param_assignment(name, obj, set(), aux_node, scope_id, definition)

    def _gen_aux_callback_for_callable(
        self, node: Union[astlib.FunctionDef, astlib.Lambda]
    ):
        param_defs = {
            d.name: d
            for d in self._definitions
            if d.enclosing_node is node and isinstance(d.node, astlib.Param)
        }
        body = node.body.body[0] if isinstance(node, astlib.FunctionDef) else node.body
        scope_id: astlib.ScopeId = self._scope_id_dict[body]

        def callback(d_locals):
            if node in self._callsite_records:
                record = self._callsite_records[node]
                binding = record["binding"]
                for arg in record["non_defaults"]:
                    self._gen_param_assignment_regular(
                        arg,
                        d_locals[arg],
                        binding.arguments[arg],
                        scope_id,
                        param_defs[arg],
                    )

                for arg in itertools.chain(
                    record["pos_var_args"], record["kw_var_args"]
                ):
                    self._gen_param_assignment_var(
                        arg,
                        d_locals[arg],
                        binding.arguments[arg],
                        scope_id,
                        param_defs[arg],
                    )

                for arg in record["defaults"]:
                    self._gen_param_assignment_defaults(
                        arg, d_locals[arg], scope_id, param_defs[arg]
                    )

            else:
                for param, param_def in param_defs.items():
                    obj = d_locals.get(param, None)
                    if obj is None:
                        continue

                    def_event = DefEvent(
                        timestamp=self.clock.get_time(),
                        owner=None,
                        name=param,
                        obj_id=id(obj),
                        scope_id=scope_id,
                        def_key=param_def,
                        ast_node=param_def.node,
                    )

                    self.trace_events_collector.add_event(def_event, obj=obj)

                self.clock.increment(step=1)

        return callback

    def _gen_aux_instrumentation_func_def(self, f_def: astlib.FunctionDef):
        callback = self._gen_aux_callback_for_callable(f_def)
        first_statement = next(astlib.iter_body_stmts(f_def.body))

        return {
            StmtCallbacksGenerator: {
                first_statement: [
                    StmtCallback(
                        callable=callback,
                        name=self.gen_stmt_callback_id(),
                        position="pre",
                        arg_str="locals()",
                    )
                ]
            }
        }

    def _gen_aux_instrumentation_lambda(self, lambda_expr: astlib.Lambda):
        callback = self._gen_aux_callback_for_callable(lambda_expr)

        return {
            ExprCallbacksGenerator: {
                lambda_expr.body: [
                    ExprCallback(
                        callable=callback,
                        name=self.gen_expr_callback_id(),
                        position="pre",
                        arg_str="locals()",
                    )
                ]
            }
        }

    def _gen_aux_instrumentation_call_expr(self, call_expr: astlib.Call):
        """
        Generates necessary instrumentation for recording auxiliary intervals for a call.
        This corresponds to explicit assignments to parameters of the function.
        :param call_expr:
        :return:
        """

        pos_args = []
        kw_args = []
        for arg in call_expr.args:
            if arg.keyword is None and arg.star != "**":
                pos_args.append(
                    {
                        "temp_var": self.temp_vars_generator.get_new_var(),
                        "node": arg.value,
                        "kind": "positional",
                        "starred": arg.star == "*",
                    }
                )
            elif arg.keyword is None and arg.star == "**":
                kw_args.append(
                    {
                        "temp_var": self.temp_vars_generator.get_new_var(),
                        "node": arg.value,
                        "kind": "keyword",
                        "starred": True,
                    }
                )
            else:
                kw_args.append(
                    {
                        "temp_var": self.temp_vars_generator.get_new_var(),
                        "node": arg.value,
                        "kind": "keyword",
                        "starred": False,
                        "name": arg.keyword.value,
                    }
                )

        expr_callbacks: Dict[
            astlib.BaseExpression, List[ExprCallback]
        ] = collections.defaultdict(list)
        expr_wrappers: Dict[
            astlib.BaseExpression, List[ExprWrapper]
        ] = collections.defaultdict(list)

        #  Expr callbacks to register assignments to temporary vars and capture the values.
        #  We capture the values to deal with the problem of lazy iterators being passed as starred arguments.
        #  We need to count the number of elements in the iterator, but for that we need to actually iterate.
        #  This messes up the arguments for the actual function call, so we patch them up before in the call decorator.
        capture_dict: Dict[str, Any] = {}
        for arg_dict in itertools.chain(pos_args, kw_args):
            expr_callbacks[arg_dict["node"]].extend(
                self._gen_arg_assignment_and_capture_callbacks(
                    arg_dict, call_expr, capture_dict
                )
            )
            expr_wrappers[arg_dict["node"]].append(
                self._gen_obj_tracker_wrapper(arg_dict["node"])
            )

        expr_wrappers[call_expr.func].append(
            self._gen_obj_tracker_wrapper(call_expr.func)
        )

        def func_aux(func, orig_pos_args, orig_kwargs):
            try:
                if func not in self._func_lambdas_encountered:
                    return orig_pos_args, orig_kwargs
            except:
                return orig_pos_args, orig_kwargs

            try:
                signature = inspect.signature(func)
            except (ValueError, TypeError):
                return orig_pos_args, orig_kwargs

            conc_pos_args = []
            sym_pos_args = []
            conc_kw_args = {}
            sym_kw_args = {}

            for arg in pos_args:
                t = arg["temp_var"]
                if arg["starred"]:
                    conc_pos_args.extend(capture_dict[t])
                    sym_pos_args.extend(
                        (t, idx) for idx, _ in enumerate(capture_dict[t])
                    )
                else:
                    conc_pos_args.append(capture_dict[t])
                    sym_pos_args.append(t)

            for arg in kw_args:
                t = arg["temp_var"]
                if arg["starred"]:
                    conc_kw_args.update(capture_dict[t])
                    sym_kw_args.update({k: (t, k) for k in capture_dict[t].keys()})
                else:
                    conc_kw_args[arg["name"]] = capture_dict[t]
                    sym_kw_args[arg["name"]] = t

            binding = signature.bind(*sym_pos_args, **sym_kw_args)
            non_defaults = set(binding.arguments.keys())
            binding.apply_defaults()
            defaults = {k for k in binding.arguments.keys() if k not in non_defaults}

            pos_var_args = {
                p
                for p in non_defaults
                if signature.parameters[p].kind == inspect.Parameter.VAR_POSITIONAL
            }
            kw_var_args = {
                p
                for p in non_defaults
                if signature.parameters[p].kind == inspect.Parameter.VAR_KEYWORD
            }

            self._callsite_records[self._func_lambdas_encountered[func]] = {
                "binding": binding,
                "non_defaults": non_defaults - pos_var_args - kw_var_args,
                "defaults": defaults,
                "pos_var_args": pos_var_args,
                "kw_var_args": kw_var_args,
            }
            return conc_pos_args, conc_kw_args

        return {
            ExprCallbacksGenerator: expr_callbacks,
            ExprWrappersGenerator: expr_wrappers,
            CallDecoratorsGenerator: {
                call_expr.func: [
                    CallDecorator(callable=func_aux, returns_new_args=True)
                ]
            },
        }

    def _gen_obj_tracker_wrapper(self, expr):
        def wrapper(value):
            self._last_known_expr_value[expr] = value
            return value

        return ExprWrapper(callable=wrapper, name=self.gen_wrapper_id())

    def _gen_arg_assignment_and_capture_callbacks(
        self, arg_dict, call_expr: astlib.Call, capture_dict
    ):
        temp_var = arg_dict["temp_var"]
        expr_node = arg_dict["node"]
        scope_id = self._scope_id_dict[expr_node]
        func_node = call_expr.func

        if not arg_dict["starred"]:
            aux_node = astlib.create_assignment([temp_var], arg_dict["node"])
            star = ""
        elif arg_dict["kind"] == "positional":
            aux_node = astlib.create_assignment(
                [temp_var], astlib.wrap_with_call(expr_node, "list")
            )
            star = "*"
        elif arg_dict["kind"] == "keyword":
            aux_node = astlib.create_assignment(
                [temp_var], astlib.wrap_with_call(expr_node, "dict")
            )
            star = "**"
        else:
            raise AssertionError

        def callback_pre():
            try:
                if (
                    self._last_known_expr_value[func_node]
                    not in self._func_lambdas_encountered
                ):
                    return
            except:
                return

            start_time = self.clock.get_time()
            self.clock.increment(step=1)

            trace_item = TraceItem(
                start_time=start_time,
                end_time=-1,
                ast_node=aux_node,
                scope_id=scope_id,
                item_type=TraceItemType.AUX,
                metadata={"aux-type": "arg_assignment"},
            )
            self.trace_items_collector.add_in_progress_item(trace_item)

        def callback_post():
            try:
                if (
                    self._last_known_expr_value[func_node]
                    not in self._func_lambdas_encountered
                ):
                    return
            except:
                return

            end_time = self.clock.increment(step=1)
            trace_item = self.trace_items_collector.get_last_in_progress_item_for_node(
                aux_node
            )
            trace_item.end_time = end_time
            obj = self._last_known_expr_value[expr_node]
            if star == "*":
                obj = list(obj)
            elif star == "**":
                obj = dict(obj)

            capture_dict[temp_var] = obj

            def_event = DefEvent(
                timestamp=end_time - 1,
                owner=trace_item,
                obj_id=id(obj),
                name=temp_var,
                scope_id=scope_id,
                def_key=temp_var,
                ast_node=trace_item.ast_node,
            )
            self.trace_events_collector.add_event(def_event, obj=obj)
            self.trace_items_collector.add_item(trace_item)

        return [
            ExprCallback(
                callable=callback_pre,
                name=self.gen_expr_callback_id(),
                arg_str="",
                position="pre",
            ),
            ExprCallback(
                callable=callback_post,
                name=self.gen_expr_callback_id(),
                arg_str="",
                position="post",
            ),
        ]


@attr.s(cmp=False, repr=False)
class ForLoopAuxGenerator(
    StmtCallbacksGenerator,
    ExprCallbacksGenerator,
    ExprWrappersGenerator,
    CallDecoratorsGenerator,
):
    """
    Auxiliary trace items plus events generator for functions and Lambdas
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()
    temp_vars_generator: TemporaryVariablesGenerator = attr.ib()

    #  Internal Stuff
    _definitions: List[astlib.Definition] = attr.ib(init=False, default=None)
    _scope_id_dict: Dict[astlib.AstNode, astlib.ScopeId] = attr.ib(
        init=False, default=None
    )
    _aux_instrumentation: Dict[Type[BaseGenerator], Dict] = attr.ib(
        init=False, default=None
    )

    def __attrs_post_init__(self):
        self.reset()

    def reset(self):
        self._scope_id_dict = {}
        self._definitions = []

    def preprocess(self, ast_root: astlib.AstNode):
        self._scope_id_dict.update(astlib.get_scope_id_mapping(ast_root))
        self._definitions.extend(astlib.get_definitions_and_accesses(ast_root)[0])

        self._aux_instrumentation = {
            StmtCallbacksGenerator: collections.defaultdict(list),
            ExprWrappersGenerator: collections.defaultdict(list),
            ExprCallbacksGenerator: collections.defaultdict(list),
            CallDecoratorsGenerator: collections.defaultdict(list),
        }

        for stmt in self.iter_stmts(ast_root):
            if isinstance(stmt, astlib.For):
                for k, v in self._gen_aux_instrumentation_for_loop(stmt).items():
                    miscutils.merge_defaultdicts_list(self._aux_instrumentation[k], v)

    def gen_stmt_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
        callbacks: Dict[
            astlib.AstStatementT, List[StmtCallback]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            callbacks, self._aux_instrumentation[StmtCallbacksGenerator]
        )
        return callbacks

    def gen_expr_wrappers(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers: Dict[
            astlib.BaseExpression, List[ExprWrapper]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            wrappers, self._aux_instrumentation[ExprWrappersGenerator]
        )
        return wrappers

    def gen_expr_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        callbacks: Dict[
            astlib.BaseExpression, List[ExprCallback]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            callbacks, self._aux_instrumentation[ExprCallbacksGenerator]
        )
        return callbacks

    def gen_decorators(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[
            astlib.BaseExpression, List[CallDecorator]
        ] = collections.defaultdict(list)

        miscutils.merge_defaultdicts_list(
            decorators, self._aux_instrumentation[CallDecoratorsGenerator]
        )
        return decorators

    def _gen_aux_instrumentation_for_loop(
        self, for_loop: astlib.For
    ) -> Dict[Type[BaseGenerator], Dict]:
        temp_var_iter: str = self.temp_vars_generator.get_new_var()
        scope_id: int = self._scope_id_dict[for_loop]
        iter_init_expr = astlib.wrap_with_call(for_loop.iter, "iter")
        aux_iter_init_node: astlib.Assign = astlib.create_assignment(
            [temp_var_iter], iter_init_expr
        )
        aux_iter_next_node: astlib.Assign = astlib.create_assignment(
            [for_loop.target], astlib.wrap_with_call(temp_var_iter, "next")
        )

        definitions = [d for d in self._definitions if d.enclosing_node is for_loop]
        temp_var_def_key = object()

        def callback_iter_init():
            iter_trace_item = self.trace_items_collector.get_last_item_for_node(
                for_loop.iter
            )
            assert iter_trace_item is not None
            end_time = self.clock.increment(step=1)

            item = TraceItem(
                start_time=iter_trace_item.start_time,
                end_time=end_time,
                ast_node=aux_iter_init_node,
                scope_id=scope_id,
                item_type=TraceItemType.AUX,
                metadata={
                    "aux-type": "for_loop_iter_init_assignment",
                },
            )
            iter_trace_item.par_item = item

            def_event = DefEvent(
                timestamp=end_time - 1,
                owner=item,
                obj_id=id(temp_var_def_key),
                name=temp_var_iter,
                scope_id=scope_id,
                def_key=temp_var_def_key,
                ast_node=item.ast_node,
            )
            self.trace_events_collector.add_event(def_event)
            self.trace_items_collector.add_item(item)

        def callback_next_iter(d_globals, d_locals):
            start_time = self.clock.get_time()
            end_time = self.clock.increment(step=2)

            item = TraceItem(
                start_time=start_time,
                end_time=end_time,
                ast_node=aux_iter_next_node,
                scope_id=scope_id,
                item_type=TraceItemType.AUX,
                metadata={
                    "aux-type": "for_loop_iter_next_assignment",
                    "targets": [d.name for d in definitions],
                },
            )

            temp_var_def_event = self.trace_events_collector.get_last_event_for_def(
                temp_var_def_key
            )
            access_event = AccessEvent(
                timestamp=start_time,
                owner=item,
                obj_id=temp_var_def_event.obj_id,
                name=temp_var_iter,
                scope_id=scope_id,
                def_event=temp_var_def_event,
                ast_node=aux_iter_next_node.value.args[0].value,
            )  # Guaranteed to be a name
            read_event = ObjReadEvent(
                timestamp=start_time,
                owner=item,
                obj_id=id(temp_var_def_key),
                ast_node=item.ast_node,
            )
            write_event = ObjWriteEvent(
                timestamp=start_time + 1,
                owner=item,
                obj_id=id(temp_var_def_key),
                ast_node=item.ast_node,
            )
            self.trace_events_collector.add_event(access_event)
            self.trace_events_collector.add_event(read_event)
            self.trace_events_collector.add_event(write_event)

            for d in definitions:
                obj = d_locals.get(d.name, d_globals.get(d.name, None))
                if obj is None:
                    continue

                def_event = DefEvent(
                    timestamp=start_time,
                    owner=item,
                    obj_id=id(obj),
                    name=d.name,
                    scope_id=scope_id,
                    def_key=d,
                    ast_node=item.ast_node,
                )
                self.trace_events_collector.add_event(def_event, obj=obj)

            self.trace_items_collector.add_item(item)

        first_statement = next(astlib.iter_body_stmts(for_loop.body))
        return {
            StmtCallbacksGenerator: {
                first_statement: [
                    StmtCallback(
                        callable=callback_next_iter,
                        name=self.gen_stmt_callback_id(),
                        position="pre",
                        arg_str="globals(), locals()",
                        priority=1,
                    )
                ]
            },
            ExprCallbacksGenerator: {
                for_loop.iter: [
                    ExprCallback(
                        callable=callback_iter_init,
                        name=self.gen_expr_callback_id(),
                        position="post",
                        arg_str="",
                    )
                ]
            },
        }
