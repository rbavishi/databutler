import collections
from abc import abstractmethod
from typing import Union, List, Dict, Any, Tuple, Optional

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessor
from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import ExprWrappersGenerator, ExprWrapper, StmtCallbacksGenerator, \
    StmtCallback, Instrumentation, Instrumenter
from databutler.pat.utils import miscutils
from databutler.utils import code as codeutils


@attrs.define(eq=True, hash=True)
class _VarAccess:
    name: str
    scope_id: int
    dtype: str
    timestamp: int
    node: astlib.AstNode = attrs.field(repr=False)


@attrs.define(eq=True, hash=True)
class _VarDef:
    name: str
    scope_id: int
    dtype: str
    timestamp: int
    node: astlib.AstNode = attrs.field(repr=False)
    enclosing_node: astlib.AstNode = attrs.field(repr=False)
    metadata: Optional[Dict] = attrs.field(eq=False, hash=False)


@attrs.define(eq=False, repr=False)
class _VarDefAndAccessTracker(StmtCallbacksGenerator, ExprWrappersGenerator):
    """
    Instrumentation generator for tracking definitions and accesses.
    """
    _time: int = 0
    events: List[Union[_VarAccess, _VarDef]] = attrs.Factory(list)

    def gen_stmt_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.AstStatementT,
                                                                   List[StmtCallback]]:
        callbacks: Dict[astlib.AstStatementT, List[StmtCallback]] = collections.defaultdict(list)

        definitions, accesses = astlib.get_definitions_and_accesses(ast_root)
        for definition in definitions:
            if isinstance(definition.node, astlib.Name) and isinstance(definition.enclosing_node,
                                                                       (astlib.Assign,
                                                                        astlib.AugAssign,
                                                                        astlib.AnnAssign)):
                #  We only care about assignments to variables, not function/class defs.
                miscutils.merge_defaultdicts_list(callbacks, self._gen_def_callbacks_assignments(definition))

        return callbacks

    def _gen_def_callbacks_assignments(self, definition: astlib.Definition):
        name: str = definition.name
        # This is the only point of difference. We consider the whole assignment to be the owner of the event.
        enc_node: astlib.AstNode = definition.enclosing_node

        def_metadata = {}

        if isinstance(enc_node, astlib.Assign):
            if isinstance(enc_node.value, astlib.Name) and len(enc_node.targets) == 1:
                #  The assignment is of the form `var1 = var2`. This allows us to perform additional optimizations.
                def_metadata['is_var_eq_var'] = True
            else:
                def_metadata['is_var_eq_var'] = False

            if astlib.is_constant(enc_node.value) and len(enc_node.targets) == 1:
                def_metadata['is_constant'] = True
                def_metadata['constant_val'] = astlib.get_constant_value(enc_node.value)
                def_metadata['constant_val_ast'] = enc_node.value
            else:
                def_metadata['is_constant'] = False
                def_metadata['constant_val'] = None
                def_metadata['constant_val_ast'] = None

        scope_id: astlib.ScopeId = definition.scope_id

        def callback(d_globals, d_locals):
            #  Get the value of the variable
            obj = d_locals.get(name, d_globals.get(name, None))
            if obj is None:
                return

            dtype = str(type(obj))
            self._time += 1
            self.events.append(
                _VarDef(
                    name=name,
                    scope_id=scope_id,
                    dtype=dtype,
                    timestamp=self._time,
                    node=definition.node,
                    enclosing_node=enc_node,
                    metadata=def_metadata,
                )
            )

        return {
            enc_node: [StmtCallback(callable=callback, name=self.gen_stmt_callback_id(),
                                    position='post', arg_str='globals(), locals()', mandatory=False)]
        }

    def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
        wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)
        definitions, accesses = astlib.get_definitions_and_accesses(ast_root)

        for access in accesses:
            if len(access.definitions) > 0 and isinstance(access.node, astlib.Name):
                #  We will wrap every name expression we find.
                wrappers[access.node].append(self._gen_access_wrapper(access))

        return wrappers

    def _gen_access_wrapper(self, access: astlib.Access):
        scope_id = access.scope_id
        definitions = access.definitions
        name = definitions[0].name

        def wrapper(value):
            #  Get the dtype, increase the time, and record an access event.
            dtype = str(type(value))
            self._time += 1
            self.events.append(
                _VarAccess(
                    name=name,
                    scope_id=scope_id,
                    dtype=dtype,
                    timestamp=self._time,
                    node=access.node,
                )
            )

            return value

        return ExprWrapper(callable=wrapper, name=self.gen_wrapper_id())


@attrs.define
class _LiveVarRange:
    """
    The primary data-structure used to find variable renamings for optimization.
    """
    #  Start timestamp, should correspond to the timestamp of the anchor def-event.
    start: int
    #  End timestamp, should correspond to the timestamp of the last access which fed off of the anchor def-event.
    end: int
    #  Name of the variable in the anchor def-event.
    name: str
    #  Scope of the variable in the anchor def-event.
    scope_id: int
    #  Data-type of the variable in the anchor def-event.
    dtype: str

    #  The anchor def-event.
    def_event: Optional[_VarDef]
    #  The accesses which semantically feed off of the anchor def-event.
    access_events: List[_VarAccess]

    #  Any other def-events which can be merged with the anchor for the purpose of renaming.
    merged_def_events: List[_VarDef]

    #  The assigned register.
    reg: int = -1


def _get_live_var_ranges(tracker: _VarDefAndAccessTracker) -> List[_LiveVarRange]:
    events = sorted(tracker.events, key=lambda x: x.timestamp)
    last_defs: Dict[Tuple[str, int], _VarDef] = {}
    range_dict: Dict[Tuple[str, int, Optional[_VarDef]], _LiveVarRange] = {}

    last_access_range = None
    last_def_for_node: Dict[astlib.AstNode, _LiveVarRange] = {}

    for event in events:
        if isinstance(event, _VarAccess):
            last_def: Optional[_VarDef] = last_defs.get((event.name, event.scope_id), None)
            range_key = (event.name, event.scope_id, last_def)

            if range_key not in range_dict:
                range_dict[range_key] = _LiveVarRange(
                    start=last_def.timestamp if last_def is not None else -1,
                    end=event.timestamp,
                    name=event.name,
                    scope_id=event.scope_id,
                    dtype=event.dtype,
                    def_event=last_def,
                    merged_def_events=[],
                    access_events=[event],
                )

            else:
                rnge = range_dict[range_key]
                rnge.end = max(rnge.end, event.timestamp)
                rnge.access_events.append(event)

            last_access_range = range_dict[range_key]

        elif isinstance(event, _VarDef):
            last_defs[event.name, event.scope_id] = event
            range_key = (event.name, event.scope_id, event)

            if event.metadata['is_var_eq_var']:
                #  We know for sure that the event recorded just before this would be the access event corresponding
                #  to the RHS.
                assert last_access_range is not None
                range_dict[range_key] = last_access_range
                last_access_range.merged_def_events.append(event)

            elif event.node in last_def_for_node:
                #  This can happen when there are loops, or calls to the same function.
                #  Since we cannot use a different variable for every iteration of the loop (we are not unrolling here),
                #  We have to change them in sync. A neat way to achieve this is by treating the accesses of this
                #  definition as if they depend on the first definition encountered in the first iteration of the loop,
                #  or the first call of the function. This in turn results in a single, bigger liveness range that
                #  in turn forces register allocation to be more conservative and accurate.
                range_dict[range_key] = rnge = last_def_for_node[event.node]
                rnge.merged_def_events.append(event)

            else:
                range_dict[range_key] = _LiveVarRange(
                    start=event.timestamp,
                    end=event.timestamp,
                    name=event.name,
                    scope_id=event.scope_id,
                    dtype=event.dtype,
                    def_event=event,
                    merged_def_events=[],
                    access_events=[],
                )

            last_def_for_node[event.node] = range_dict[range_key]

        else:
            raise TypeError(f"Unrecognized event of type {type(event)}")

    return sorted(range_dict.values(), key=lambda x: x.start)


def _perform_register_allocation(ranges: List[_LiveVarRange]) -> None:
    #  Use the linear-scan algorithm to perform greedy register allocation as if we have infinite registers.
    registers = set(range(0, len(ranges)))
    all_ranges = sorted(ranges, key=lambda x: x.start)
    reg_to_compat_key: Dict[int, Tuple[int, str]] = {}
    active = set()

    for idx, rnge in enumerate(all_ranges):
        #  Expire old ranges and free up registers
        for a_idx in list(active):
            active_range = all_ranges[a_idx]
            if active_range.end < rnge.start:
                active.remove(a_idx)
                registers.add(active_range.reg)

        #  Assign one of the available registers
        compat_key = (rnge.scope_id, rnge.dtype)
        for free_reg in sorted(registers):
            if free_reg not in reg_to_compat_key or reg_to_compat_key[free_reg] == compat_key:
                registers.remove(free_reg)
                rnge.reg = free_reg
                reg_to_compat_key[free_reg] = compat_key
                active.add(idx)
                break
        else:
            raise AssertionError("Free register not found")


def _rename_variables(code_ast: astlib.AstNode, tracker: _VarDefAndAccessTracker):
    #  First, compute the liveness ranges of variables.
    ranges = _get_live_var_ranges(tracker)

    #  Run register-allocation to get groups of defs and accesses that can share a single variable name.
    _perform_register_allocation(ranges)

    #  Create a node-replacement map based on the register allocation
    reg_assignment_dict: Dict[int, List[_LiveVarRange]] = collections.defaultdict(list)
    for rnge in ranges:
        reg_assignment_dict[rnge.reg].append(rnge)

    repl_map = {}
    for reg, reg_ranges in reg_assignment_dict.items():
        new_name = reg_ranges[0].name
        for rnge in reg_ranges:
            if rnge.def_event is not None and rnge.def_event.metadata['is_constant']:
                constant_val = rnge.def_event.metadata['constant_val']
                constant_val_ast = rnge.def_event.metadata['constant_val_ast']

                if (not isinstance(constant_val, (list, tuple, set, dict))) or len(rnge.access_events) < 3:
                    #  Remove the assignments.
                    repl_map[rnge.def_event.enclosing_node] = None

                    for merged_def in rnge.merged_def_events:
                        repl_map[merged_def.enclosing_node] = None

                    #  Replace the assignments by the constant value
                    for access in rnge.access_events:
                        repl_map[access.node] = constant_val_ast

                    continue

            if rnge.def_event is not None:
                repl_map[rnge.def_event.node] = astlib.create_name_expr(new_name)

            for merged_def in rnge.merged_def_events:
                repl_map[merged_def.node] = astlib.create_name_expr(new_name)

            for access in rnge.access_events:
                repl_map[access.node] = astlib.create_name_expr(new_name)

    #  Perform the renaming.
    new_ast = astlib.with_deep_replacements(code_ast, repl_map)
    #  Return after removing unnecessary code and running formatting.
    return codeutils.optimize_code(astlib.to_code(new_ast))


@attrs.define(eq=False, repr=False)
class VarNameOptimizer(DatanaFunctionProcessor):
    """
    A processor that optimizes code by removing unnecessary variable names and reusing existing ones.
    """

    def _process(self, d_func: DatanaFunction) -> DatanaFunction:
        code = d_func.code_str
        #  Set up instrumentation.
        tracker = _VarDefAndAccessTracker()

        normalizer_instrumentation = Instrumentation.from_generators(tracker)
        instrumenter = Instrumenter(normalizer_instrumentation)

        #  We use PAT's wrappers for ASTs as it is more expressive and powerful.
        #  The instrumentation library also relies on it.
        code_ast = astlib.parse(code)
        inst_ast, global_ctx = instrumenter.process(code_ast)
        inst_code = astlib.to_code(inst_ast)

        #  Execute the code as per the client domain's requirements.
        #  Once the instrumented code is run, the finder should have populated its
        #  internal data-structures for us to use.
        self._run_function_code(func_code=inst_code, func_name=d_func.func_name,
                                pos_args=d_func.get_pos_args() or [],
                                kw_args=d_func.get_kw_args() or {},
                                global_ctx=global_ctx)

        new_code = _rename_variables(code_ast, tracker)

        #  Assemble the result
        new_d_func = d_func.copy()
        new_d_func.code_str = new_code
        new_d_func.metadata = new_d_func.metadata or {}
        new_d_func.metadata[self.get_processor_metadata_key()] = {
            "old_code": d_func.code_str,
        }

        return new_d_func

    @classmethod
    def get_processor_name(cls) -> str:
        return "var-name-optimizer"

    @abstractmethod
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        """
        Runs the provided function with the given args and global context.

        Must be implemented by every class implementing FuncNameExtractor.

        Args:
            func_code: A string corresponding to the function to be executed.
            func_name: The name of the function as a string.
            pos_args: A list of positional arguments to be provided to the function.
            kw_args: A dictionary of keyword arguments to be provided to the function.
            global_ctx: The global context in which to run the function.
        """
