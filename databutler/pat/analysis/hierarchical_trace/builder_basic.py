import collections
import itertools
from typing import Dict, List, Union, Set

import attr

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace import specutils
from databutler.pat.analysis.hierarchical_trace.builder_utils import (
    TraceEventsCollector,
    TraceItemsCollector,
    iter_potentially_modified_subexprs,
)
from databutler.pat.analysis.hierarchical_trace.core import (
    DefEvent,
    AccessEvent,
    ObjReadEvent,
    ObjWriteEvent,
)
from databutler.pat.analysis.instrumentation import (
    StmtCallbacksGenerator,
    StmtCallback,
    ExprCallbacksGenerator,
    ExprWrappersGenerator,
    ExprWrapper,
    ExprCallback,
    CallDecoratorsGenerator,
    CallDecorator,
)
from databutler.pat.utils import miscutils
from databutler.utils.logging import logger


@attr.s(cmp=False, repr=False)
class BasicDefEventsGenerator(StmtCallbacksGenerator):
    """
    Handles generation of basic def events that do not require auxiliary trace items.
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()

    def gen_stmt_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
        callbacks: Dict[
            astlib.AstStatementT, List[StmtCallback]
        ] = collections.defaultdict(list)

        definitions, accesses = astlib.get_definitions_and_accesses(ast_root)
        for definition in definitions:
            if astlib.is_stmt(definition.node):
                miscutils.merge_defaultdicts_list(
                    callbacks, self._gen_def_callbacks_simple(definition)
                )

            elif isinstance(definition.node, astlib.Name) and isinstance(
                definition.enclosing_node,
                (astlib.Assign, astlib.AugAssign, astlib.AnnAssign),
            ):
                miscutils.merge_defaultdicts_list(
                    callbacks, self._gen_def_callbacks_assignments(definition)
                )

        return callbacks

    def _gen_def_callbacks_simple(self, definition: astlib.Definition):
        name: str = definition.name
        node: astlib.AstNode = definition.node
        scope_id: astlib.ScopeId = definition.scope_id

        def callback(d_globals, d_locals):
            #  Find the object this name is currently mapped to.
            obj = d_locals.get(name, d_globals.get(name, None))
            if obj is None:
                return

            #  Get the trace item to associate the def event with.
            trace_item = self.trace_items_collector.get_last_item_for_node(node)
            if trace_item is None:
                return

            def_event = DefEvent(
                timestamp=trace_item.end_time - 1,
                owner=trace_item,
                name=name,
                obj_id=id(obj),
                scope_id=scope_id,
                def_key=definition,
                ast_node=trace_item.ast_node,
            )

            self.trace_events_collector.add_event(def_event, obj=obj)

        return {
            node: [
                StmtCallback(
                    callable=callback,
                    name=self.gen_stmt_callback_id(),
                    position="post",
                    arg_str="globals(), locals()",
                    mandatory=False,
                )
            ]
        }

    def _gen_def_callbacks_assignments(self, definition: astlib.Definition):
        name: str = definition.name
        # This is the only point of difference. We consider the whole assignment to be the owner of the event.
        node: astlib.AstNode = definition.enclosing_node
        scope_id: astlib.ScopeId = definition.scope_id

        def callback(d_globals, d_locals):
            obj = d_locals.get(name, d_globals.get(name, None))
            if obj is None:
                return

            trace_item = self.trace_items_collector.get_last_item_for_node(node)
            if trace_item is None:
                return

            def_event = DefEvent(
                timestamp=trace_item.end_time - 1,
                owner=trace_item,
                name=name,
                obj_id=id(obj),
                scope_id=scope_id,
                def_key=definition,
                ast_node=trace_item.ast_node,
            )

            self.trace_events_collector.add_event(def_event, obj=obj)

        return {
            node: [
                StmtCallback(
                    callable=callback,
                    name=self.gen_stmt_callback_id(),
                    position="post",
                    arg_str="globals(), locals()",
                    mandatory=False,
                )
            ]
        }


@attr.s(cmp=False, repr=False)
class BasicAccessEventsGenerator(ExprWrappersGenerator, ExprCallbacksGenerator):
    """
    Handles generation of access events.
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()

    #  Internal Stuff
    _accesses: List[astlib.Access] = attr.ib(init=False, default=None)

    def preprocess(self, ast_root: astlib.AstNode):
        _, self._accesses = astlib.get_definitions_and_accesses(ast_root)

    def gen_expr_wrappers(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers: Dict[
            astlib.BaseExpression, List[ExprWrapper]
        ] = collections.defaultdict(list)

        for access in self._accesses:
            wrappers[access.node].append(self._gen_access_wrapper(access))

        return wrappers

    def gen_expr_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        callbacks: Dict[
            astlib.BaseExpression, List[ExprCallback]
        ] = collections.defaultdict(list)

        for stmt in astlib.iter_stmts(ast_root):
            if isinstance(stmt, astlib.AugAssign) and isinstance(
                stmt.target, astlib.Name
            ):
                #  The access corresponding to this target will not be covered by the previous step as its
                #  expression context will be a store, and hence the wrapper will be  discarded by the
                #  instrumentation (with a warning). Hence we need to handle this specially.
                assert any(stmt.target is a.node for a in self._accesses)
                miscutils.merge_defaultdicts_list(
                    callbacks, self._gen_access_callbacks_aug_assign(stmt)
                )

        return callbacks

    def _gen_access_wrapper(self, access: astlib.Access):
        node = access.node
        scope_id = access.scope_id
        definitions = access.definitions

        def wrapper(value):
            """
            Get the in-progress trace item for this node, and use it to generate an access event.
            :param value:
            :return:
            """
            obj_id: int = id(value)
            def_events = list(
                filter(
                    None,
                    [
                        self.trace_events_collector.get_last_event_for_def(d)
                        for d in definitions
                    ],
                )
            )
            def_events = sorted(def_events, key=lambda x: -x.timestamp)

            trace_item = self.trace_items_collector.get_last_in_progress_item_for_node(
                node
            )
            if trace_item is None:
                logger.warning(f"Did not find trace item for node of type {type(node)}")
                return value

            try:
                #  Find the last definition to be recorded that is relevant for this access.
                def_event = next(d for d in def_events if d.obj_id == obj_id)

                event = AccessEvent(
                    timestamp=self.clock.get_time(),
                    owner=trace_item,
                    name=def_event.name,
                    scope_id=scope_id,
                    obj_id=obj_id,
                    def_event=def_event,
                    ast_node=trace_item.ast_node,
                )

                self.trace_events_collector.add_event(event, obj=value)

            except StopIteration:
                # logger.warning(f"Did not find matching def event for access {astlib.to_code(node)}.")
                pass

            return value

        return ExprWrapper(callable=wrapper, name=self.gen_wrapper_id())

    def _gen_access_callbacks_aug_assign(self, stmt: astlib.AugAssign):
        assert isinstance(stmt.target, astlib.Name)
        access = next(a for a in self._accesses if a.node is stmt.target)
        scope_id = access.scope_id
        definitions = access.definitions
        code_str = astlib.to_code(stmt.target)

        def callback(value):
            """
            Get the in-progress trace item for the aug-assign, and use it to generate an access event.
            :param value:
            :return:
            """
            #  The argument to this callback is the value of the target (see arg_str below).
            #  This only works if evaluating the target is non side-effecting.
            obj_id: int = id(value)
            def_events = list(
                filter(
                    None,
                    [
                        self.trace_events_collector.get_last_event_for_def(d)
                        for d in definitions
                    ],
                )
            )
            def_events = sorted(def_events, key=lambda x: -x.timestamp)

            trace_item = self.trace_items_collector.get_last_in_progress_item_for_node(
                stmt
            )
            if trace_item is None:
                logger.warning(f"Did not find trace item for node of type {type(stmt)}")
                return

            try:
                #  Find the last definition to be recorded that is relevant for this access.
                def_event = next(d for d in def_events if d.obj_id == obj_id)

                event = AccessEvent(
                    timestamp=self.clock.get_time(),
                    owner=trace_item,
                    name=def_event.name,
                    scope_id=scope_id,
                    obj_id=obj_id,
                    def_event=def_event,
                    ast_node=stmt.target,
                )

                self.trace_events_collector.add_event(event, obj=value)

            except StopIteration:
                # logger.warning(f"Did not find matching def event for access.")
                return

        return {
            stmt.value: [
                ExprCallback(
                    callable=callback,
                    name=self.gen_expr_callback_id(),
                    arg_str=code_str,
                    position="post",
                )
            ]
        }


@attr.s(cmp=False, repr=False)
class BasicReadEventsGenerator(ExprCallbacksGenerator, ExprWrappersGenerator):
    """
    Handles generation of basic read events.
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()

    def gen_expr_wrappers(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers: Dict[
            astlib.BaseExpression, List[ExprWrapper]
        ] = collections.defaultdict(list)

        for expr in self.iter_valid_exprs(ast_root):
            wrappers[expr].append(self._gen_read_wrapper(expr))

        return wrappers

    def gen_expr_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        callbacks: Dict[
            astlib.BaseExpression, List[ExprCallback]
        ] = collections.defaultdict(list)

        for stmt in astlib.iter_stmts(ast_root):
            if isinstance(stmt, astlib.AugAssign):
                miscutils.merge_defaultdicts_list(
                    callbacks, self._gen_read_callbacks_aug_assign(stmt)
                )

        return callbacks

    def _gen_read_wrapper(self, expr: astlib.BaseExpression):
        def wrapper(value):
            obj_id = id(value)
            trace_item = self.trace_items_collector.get_last_in_progress_item_for_node(
                expr
            )
            if trace_item is None:
                return value

            event = ObjReadEvent(
                timestamp=self.clock.get_time(),
                owner=trace_item,
                obj_id=obj_id,
                ast_node=trace_item.ast_node,
            )

            self.trace_events_collector.add_event(event)
            return value

        return ExprWrapper(callable=wrapper, name=self.gen_expr_callback_id())

    def _gen_read_callbacks_aug_assign(self, aug: astlib.AugAssign):
        #  Record the value of the target after RHS is evaluated and use it to create a read event.
        #  This only works if evaluating the target is non side-effecting.
        target = aug.target
        code_str = astlib.to_code(target)

        def callback(value):
            obj_id = id(value)
            trace_item = self.trace_items_collector.get_last_in_progress_item_for_node(
                aug
            )
            if trace_item is None:
                return value

            event = ObjReadEvent(
                timestamp=self.clock.get_time(),
                owner=trace_item,
                obj_id=obj_id,
                ast_node=trace_item.ast_node,
            )

            self.trace_events_collector.add_event(event)

        return {
            aug.value: [
                ExprCallback(
                    callable=callback,
                    name=self.gen_expr_callback_id(),
                    arg_str=code_str,
                    position="post",
                )
            ]
        }


@attr.s(cmp=False, repr=False)
class BasicWriteEventsGenerator(StmtCallbacksGenerator, ExprWrappersGenerator):
    """
    Handles generation of basic write events.
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()

    def gen_stmt_callbacks(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
        callbacks: Dict[
            astlib.AstStatementT, List[StmtCallback]
        ] = collections.defaultdict(list)

        for stmt in self.iter_stmts(ast_root):
            if isinstance(stmt, (astlib.Assign, astlib.AugAssign, astlib.AnnAssign)):
                miscutils.merge_defaultdicts_list(
                    callbacks, self._gen_write_callbacks_assignments(ast_root, stmt)
                )

        return callbacks

    def _gen_write_callbacks_assignments(
        self,
        ast_root: astlib.AstNode,
        stmt: Union[astlib.Assign, astlib.AugAssign, astlib.AnnAssign],
    ) -> Dict[astlib.AstStatementT, List[StmtCallback]]:

        targets: List[astlib.BaseAssignTargetExpression] = []
        if isinstance(stmt, astlib.Assign):
            targets.extend(t.target for t in stmt.targets)

        elif isinstance(stmt, (astlib.AnnAssign, astlib.AugAssign)):
            targets.append(stmt.target)

        else:
            raise TypeError(f"Unrecognized assignment type {type(stmt)}.")

        #  We record a write to the result of evaluating all expressions on the LHS.
        #  For example, a.b = 10 would record a write for the object corresponding to `a`.
        modified_exprs = set()
        for t in targets:
            modified_exprs.update(iter_potentially_modified_subexprs(t))

        valid_exprs = set()
        for t in targets:
            valid_exprs.update(self.iter_valid_exprs(t, ast_root))

        modified_exprs.intersection_update(valid_exprs)

        def callback():
            trace_item = self.trace_items_collector.get_last_item_for_node(stmt)

            for expr in modified_exprs:
                obj_id = self.trace_items_collector.get_last_item_for_node(expr).obj_id
                assert obj_id is not None

                event = ObjWriteEvent(
                    timestamp=trace_item.end_time - 1,
                    owner=trace_item,
                    obj_id=obj_id,
                    ast_node=trace_item.ast_node,
                )

                self.trace_events_collector.add_event(event)

        return {
            stmt: [
                StmtCallback(
                    callable=callback,
                    name=self.gen_stmt_callback_id(),
                    position="post",
                    mandatory=False,
                    arg_str="",
                )
            ]
        }


@attr.s(cmp=False, repr=False)
class FunctionCallSpecsChecker(CallDecoratorsGenerator):
    """
    For function calls, check for side-effects using specs if available.
    """

    clock: LogicalClock = attr.ib()
    trace_events_collector: TraceEventsCollector = attr.ib()
    trace_items_collector: TraceItemsCollector = attr.ib()

    def __attrs_post_init__(self):
        specutils.setup(
            clock=self.clock,
            trace_events_collector=self.trace_events_collector,
            trace_items_collector=self.trace_items_collector,
        )

    def gen_decorators(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[
            astlib.BaseExpression, List[CallDecorator]
        ] = collections.defaultdict(list)

        for n in self.iter_calls(ast_root):
            miscutils.merge_defaultdicts_list(
                decorators, self.gen_spec_checker(n, ast_root=ast_root)
            )

        return decorators

    def gen_spec_checker(
        self, call: astlib.Call, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        """
        We will replace the func node with a decorator of sorts that will consult rules, if any
        and determine if the function is side-effecting.
        :param call:
        :param ast_root:
        :return:
        """

        #  Now here's the problem. If the arguments are attributes, or indices of an array or something
        #  of that sort, in an ideal setting, if you access the containing object later, you have to mark a
        #  read for the attribute/index being accessed over here, and that's how you establish the dependency.
        #  But when you have libraries like pandas which do all sorts of things under the hood, you cannot
        #  rely on the attributes being the same. A data-frame for example, in the course of some other operation
        #  is highly likely to change the objects representing the columns for example. So if a series
        #  was being modified in this particular call, the dependency would be lost further in time.

        #  So we try to be extra conservative here itself. We not only mark a write for the argument object,
        #  but for the objects via which it was accessed. This may not bode well for precision, but we really
        #  need to be sound for data-frames.

        #  Still to be a bit more precise, we treat this as an assignment. So we see if the corresponding
        #  AST node is eligible to have an ast.Store context. If yes, we do what we did for assignments.

        #  TODO : Handle starred args

        #  TODO : Can we do byte-code level instrumentation to record writes and post-morten associate them with
        #  TODO : appropriate trace item?

        modified_exprs_map: Dict[astlib.AstNode, Set[astlib.AstNode]] = {}
        for root in itertools.chain(
            [call.func], (arg.value for arg in call.args if arg.star == "")
        ):
            for node in astlib.iter_true_exprs(root, ast_root):
                modified_exprs_map[node] = set(iter_potentially_modified_subexprs(node))

        def checker(func, ret_val, args, kwargs):
            if specutils.ignore_func(func, ret_val, args, kwargs):
                return

            if not specutils.has_spec(func, ret_val, args, kwargs):
                #  Be conservative and mark writes for all the arguments.
                reads_and_writes = [
                    (o, "write") for o in itertools.chain(args, kwargs.values())
                ]
            else:
                reads_and_writes = specutils.check_spec(func, ret_val, args, kwargs)

            #  Convert the UIDs to the objects they last evaluated to.
            extended_writes_map = {}
            for node, modified_subexprs in modified_exprs_map.items():
                modified_subexprs.discard(node)

                key = self.trace_items_collector.get_last_item_for_node(node).obj_id
                ext_writes = []
                for u in modified_subexprs:
                    last_item = self.trace_items_collector.get_last_item_for_node(u)
                    if last_item is not None:
                        ext_writes.append(last_item.obj_id)

                extended_writes_map[key] = ext_writes

            if len(reads_and_writes) > 0:
                call_trace_item = (
                    self.trace_items_collector.get_last_in_progress_item_for_node(call)
                )
                for obj, event_type in reads_and_writes:
                    if event_type == "read":
                        event = ObjReadEvent(
                            timestamp=self.clock.get_time(),
                            owner=call_trace_item,
                            obj_id=id(obj),
                            ast_node=call_trace_item.ast_node,
                        )
                        self.clock.increment(1)
                        self.trace_events_collector.add_event(event)

                    else:
                        for written_obj_id in itertools.chain(
                            [id(obj)], extended_writes_map.get(id(obj), [])
                        ):
                            event = ObjWriteEvent(
                                timestamp=self.clock.get_time(),
                                owner=call_trace_item,
                                obj_id=written_obj_id,
                                ast_node=call_trace_item.ast_node,
                            )
                            self.clock.increment(1)
                            self.trace_events_collector.add_event(event)

        return {
            call.func: [
                CallDecorator(
                    callable=checker, does_not_return=True, needs_return_value=True
                )
            ]
        }
