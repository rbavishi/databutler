import collections
import os
from typing import Set, Dict, List, Tuple, Iterator, Optional, Any

import attrs
import yaml

import pandas as pd
from databutler.mining.kaggle.dynamic_analysis.instrumentation_utils import IPythonMagicBlocker
from databutler.mining.kaggle.execution.base import BaseExecutor, register_runner
from databutler.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType
from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation, \
    HierarchicalTraceInstrumentationHooks
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, TraceItem, ObjWriteEvent, DefEvent, \
    AccessEvent, DependencyType
from databutler.pat.analysis.instrumentation import Instrumentation, Instrumenter, ExprWrappersGenerator, ExprWrapper
from databutler.utils import inspection
from databutler.utils.logging import logger


@attrs.define(eq=False, repr=False)
class FuncModNameFinder(ExprWrappersGenerator):
    _func_calls_to_name: Dict[astlib.Call, str] = attrs.field(init=False, factory=dict)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                yield expr.func, ExprWrapper(
                    callable=self._gen_func_mod_name_finding_wrapper(expr),
                    name=self.gen_wrapper_id(),
                )

    def _gen_func_mod_name_finding_wrapper(self, call_expr: astlib.Call):
        def wrapper(value):
            qual_name = inspection.get_qualified_module_name(value)
            if qual_name is not None:
                self._func_calls_to_name[call_expr] = qual_name

            return value

        return wrapper

    def get_func_calls_to_names(self) -> Dict[astlib.Call, str]:
        return self._func_calls_to_name.copy()


@attrs.define(eq=False, repr=False)
class DfCollector(ExprWrappersGenerator):
    clock: LogicalClock
    trace_hooks: HierarchicalTraceInstrumentationHooks

    _read_csv_dfs: Dict[astlib.Call, pd.DataFrame] = attrs.field(init=False, factory=dict)
    _df_exprs: Dict[astlib.BaseExpression, Set[int]] = attrs.field(init=False,
                                                                   factory=lambda: collections.defaultdict(set))
    _df_obj_refs: Dict[int, pd.DataFrame] = attrs.field(init=False, factory=dict)
    _df_obj_store: Dict[int, Dict[int, pd.DataFrame]] = attrs.field(init=False,
                                                                    factory=lambda: collections.defaultdict(dict))

    def __attrs_post_init__(self):
        self.trace_hooks.install_event_handler(ObjWriteEvent, self.df_obj_write_event_handler)

    def df_obj_write_event_handler(self, event: ObjWriteEvent, **kwargs):
        if event.obj_id in self._df_obj_store:
            cur_time = self.clock.get_time()
            self._df_obj_store[event.obj_id][cur_time] = self._df_obj_refs[event.obj_id].copy()

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        #  First handle the read_csv calls
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call) and "read_csv" in astlib.to_code(expr.func):
                yield expr, ExprWrapper(
                    callable=self._gen_read_csv_wrapper(expr),
                    name=self.gen_wrapper_id(),
                )

        #  Trap any expressions producing dataframes. This will keep track of all the dataframe objects seen during
        #  execution.
        for expr in self.iter_valid_exprs(ast_root):
            yield expr, ExprWrapper(
                callable=self._gen_df_collecting_wrapper(expr),
                name=self.gen_wrapper_id(),
            )

    def _gen_read_csv_wrapper(self, expr: astlib.Call):
        def wrapper(value):
            if isinstance(value, pd.DataFrame):
                self._read_csv_dfs[expr] = value.copy()

            return value

        return wrapper

    def get_read_csv_dfs(self) -> Dict[astlib.Call, pd.DataFrame]:
        return self._read_csv_dfs.copy()

    def _gen_df_collecting_wrapper(self, expr: astlib.BaseExpression):
        def wrapper(value):
            if isinstance(value, pd.DataFrame):
                self._df_exprs[expr].add(id(value))
                if id(value) not in self._df_obj_refs:
                    self._df_obj_refs[id(value)] = value
                    self._df_obj_store[id(value)][self.clock.get_time()] = value.copy()

            return value

        return wrapper

    def get_df_exprs(self) -> Set[astlib.BaseExpression]:
        return set(self._df_exprs.keys())

    def get_df_obj_ids(self) -> Set[int]:
        return set(self._df_obj_refs.keys())

    def get_df_snapshot(self, obj_id: int, latest_before: int) -> Optional[pd.DataFrame]:
        for timestamp, df in sorted(self._df_obj_store[obj_id].items(), key=lambda x: -x[0]):
            if timestamp < latest_before:
                return df

        print(f"Did not find: {obj_id} {latest_before} {list(self._df_obj_store[obj_id].keys())}")


@attrs.define(eq=False, repr=False)
class NotebookPrintCollector(ExprWrappersGenerator):
    _expr_stmts: Set[astlib.Expr] = attrs.field(init=False, factory=set)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for node in astlib.walk(ast_root):
            if isinstance(node, astlib.NotebookCell):
                body_stmts = list(astlib.iter_body_stmts(node.body))
                if len(body_stmts) == 0:
                    continue

                last_body_stmt = body_stmts[-1]
                if isinstance(last_body_stmt, astlib.Expr):
                    yield last_body_stmt.value, ExprWrapper(
                        callable=self._gen_collector(last_body_stmt),
                        name=self.gen_wrapper_id()
                    )

    def _gen_collector(self, expr_stmt: astlib.Expr):
        def wrapper(value):
            if value is not None:
                self._expr_stmts.add(expr_stmt)

            return value

        return wrapper

    def get_expr_stmts(self) -> Set[astlib.Expr]:
        return self._expr_stmts.copy()


@attrs.define(eq=False, repr=False)
class DfStrColumnsCollector(ExprWrappersGenerator):
    #  Internal
    _collected_cols: Set[str] = attrs.field(init=False, factory=set)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            if not astlib.is_constant(expr):
                yield expr, ExprWrapper(
                    callable=self._collector,
                    name=self.gen_wrapper_id(),
                )

    def _collector(self, value):
        if isinstance(value, pd.DataFrame):
            if value.columns.nlevels == 1:
                self._collected_cols.update(c for c in value.columns if isinstance(c, str))
            else:
                flattened = sum((list(i) for i in value.columns.levels), [])
                self._collected_cols.update(c for c in flattened if isinstance(c, str))

        return value

    def get_collected_cols(self) -> Set[str]:
        return self._collected_cols.copy()


@attrs.define(eq=False, repr=False)
class DfColAttrAccessCollector(ExprWrappersGenerator):
    #  Internal
    _collected_accesses: Dict[astlib.Attribute, str] = attrs.field(init=False, factory=dict)
    _df_exprs: Set[astlib.BaseExpression] = attrs.field(init=False, factory=set)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in astlib.walk(ast_root):
            if isinstance(expr, astlib.Attribute):
                has_store_ctx = astlib.expr_has_store_ctx(expr, ast_root)
                yield expr.value, ExprWrapper(
                    callable=self._gen_collector(expr, has_store_ctx),
                    name=self.gen_wrapper_id(),
                )

    def _gen_collector(self, expr: astlib.Attribute, has_store_ctx: bool):
        attr_name = expr.attr.value

        def wrapper(value):
            try:
                if isinstance(value, pd.DataFrame) and attr_name.isidentifier():
                    if has_store_ctx or isinstance(getattr(value, attr_name), pd.Series):
                        self._collected_accesses[expr] = attr_name
            except:
                pass

            return value

        return wrapper

    def get_df_col_attr_accesses(self) -> Dict[astlib.Attribute, str]:
        return self._collected_accesses.copy()


@attrs.define(eq=False, repr=False)
class PandasMiner(BaseExecutor):
    @classmethod
    @register_runner(name="pandas_runner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        #  A clock is critical in identifying dependencies.
        clock = LogicalClock()
        #  Trace instrumentation does the heavy-lifting of recording reads/writes, var. defs and their uses.
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        #  Ready up instrumentation to track function names (and identify which ones are pandas functions)
        func_mod_name_finder = FuncModNameFinder()
        #  Ready up the instrumentation for the df detectors.
        df_collector = DfCollector(clock, trace_instrumentation.get_hooks())
        col_collector = DfStrColumnsCollector()
        #  Collect last stmts of notebook cells that evaluate to something that is printed
        nb_print_expr_collector = NotebookPrintCollector()
        #  Need to avoid executing matplotlib magics as they can mess with the config.
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        name_finding_instrumentation = Instrumentation.from_generators(func_mod_name_finder)
        df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        col_collector_instrumentation = Instrumentation.from_generators(col_collector)
        magic_blocker_instrumentation = Instrumentation.from_generators(magic_blocker)
        nb_print_expr_collector_instrumentation = Instrumentation.from_generators(nb_print_expr_collector)

        #  Merge all the instrumentation together.
        instrumentation = (trace_instrumentation |
                           name_finding_instrumentation |
                           df_collector_instrumentation |
                           col_collector_instrumentation |
                           nb_print_expr_collector_instrumentation |
                           magic_blocker_instrumentation)

        instrumenter = Instrumenter(instrumentation)

        #  Parse the source as an AST.
        if source_type == KaggleNotebookSourceType.IPYTHON_NOTEBOOK:
            code_ast = astlib.parse(source, extension='.ipynb')
        elif source_type == KaggleNotebookSourceType.PYTHON_SOURCE_FILE:
            code_ast = astlib.parse(source)
        else:
            raise NotImplementedError(f"Could not recognize source of type {source_type}")

        #  Run the instrumenter, and execute.
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        logger.info("Starting Execution Now.")
        try:
            exec(new_code, globs, globs)
        except BaseException as e:
            #  Even if it fails, we want to use whatever we get.
            logger.warning("Execution failed.")
            logger.exception(e)

        logger.info("Finished Execution.")

        #  Since we added the tracing instrumentation, we will now we able to extract the trace of the program.
        #  This trace contains all the information we need to extract dependencies, and do things like slicing.
        trace = trace_instrumentation.get_hierarchical_trace()

        #  Use the trace, matplotlib figure and df detectors to extract visualization code.
        cls._extract_pandas_code(code_ast, trace,
                                 func_mod_name_finder=func_mod_name_finder,
                                 df_collector=df_collector, col_collector=col_collector,
                                 nb_print_expr_collector=nb_print_expr_collector,
                                 output_dir_path=output_dir_path)

    @classmethod
    def _extract_pandas_code(cls, code_ast: astlib.AstNode,
                             trace: HierarchicalTrace,
                             func_mod_name_finder: FuncModNameFinder,
                             df_collector: DfCollector,
                             col_collector: DfStrColumnsCollector,
                             nb_print_expr_collector: NotebookPrintCollector,
                             output_dir_path: str):

        func_mod_name_map: Dict[astlib.Call, str] = func_mod_name_finder.get_func_calls_to_names()
        found: List[Dict] = []

        #  We want to use the top-level statements of the notebook in the extracted code, so we compute and keep aside.
        body_stmts = set()
        for s in astlib.iter_body_stmts(code_ast):
            if isinstance(s, astlib.NotebookCell):
                body_stmts.update(astlib.iter_body_stmts(s.body))
            else:
                body_stmts.add(s)

        allowed_slicing_stmts = set(body_stmts)
        for read_csv_expr, df in df_collector.get_read_csv_dfs().items():
            for parent in astlib.iter_parents(read_csv_expr, context=code_ast):
                if parent in body_stmts:
                    allowed_slicing_stmts.discard(parent)
                    break

        df_obj_ids: Set[int] = df_collector.get_df_obj_ids()
        obj_id_to_writing_items: Dict[int, List[TraceItem]] = collections.defaultdict(list)
        for e in trace.get_events():
            if isinstance(e, ObjWriteEvent) and e.obj_id in df_obj_ids:
                item = e.owner
                if item.ast_node in body_stmts:
                    obj_id_to_writing_items[e.obj_id].append(item)
                else:
                    for i in trace.iter_parents(item):
                        if i.ast_node in body_stmts:
                            obj_id_to_writing_items[e.obj_id].append(i)
                            break

        disallowed_dependencies: Set[astlib.AstNode] = set()
        for v in obj_id_to_writing_items.values():
            disallowed_dependencies.update(i.ast_node for i in v)

        for read_csv_expr, df in df_collector.get_read_csv_dfs().items():
            for parent in astlib.iter_parents(read_csv_expr, context=code_ast):
                if parent in body_stmts:
                    disallowed_dependencies.add(parent)

        criteria_items = set()
        for df_writing_items in obj_id_to_writing_items.values():
            criteria_items.update((item, "DF_WRITE") for item in df_writing_items)

        df_exprs = df_collector.get_df_exprs()
        print_exprs = nb_print_expr_collector.get_expr_stmts()
        for item in trace.items:
            if item.ast_node in print_exprs:
                all_sub_nodes = set(astlib.walk(item.ast_node))
                if not all_sub_nodes.isdisjoint(df_exprs):
                    criteria_items.add((item, "PRINT_EXPR"))

        def_event_to_accesses = collections.defaultdict(list)
        for event in trace.get_events():
            if isinstance(event, AccessEvent) and event.def_event is not None:
                def_event_to_accesses[event.def_event].append(event)

        for item in trace.items:
            if item.ast_node in allowed_slicing_stmts:
                if isinstance(item.ast_node, astlib.Assign) and len(item.ast_node.targets) == 1:
                    if item.ast_node.value in df_exprs:
                        #  The variable must be used more than once
                        def_event = None
                        for event in trace.get_events(start_time=item.start_time, end_time=item.end_time):
                            if isinstance(event, DefEvent):
                                def_event = event
                                break

                        if def_event is not None and len(def_event_to_accesses[def_event]) > 1:
                            obj_id_to_writing_items[def_event.obj_id].append(item)
                            for c_item in trace.iter_children(item):
                                if c_item.ast_node is item.ast_node.value:
                                    criteria_items.add((c_item, "DF_ASSIGN"))
                                    break

        df_obj_id_to_pkl_paths: Dict[int, str] = {}
        df_obj_id_to_df: Dict[int, pd.DataFrame] = {}
        for item, snippet_type in criteria_items:
            try:
                criteria = {item}

                ignore_timestamps = {}
                for obj_id, w_items in obj_id_to_writing_items.items():
                    for i in sorted(w_items, key=lambda x: -x.end_time):
                        if i.end_time <= item.start_time:
                            ignore_timestamps[obj_id] = i.end_time
                            break

                viz_slice: List[TraceItem] = cls._get_slice(trace, criteria, allowed_slicing_stmts,
                                                            ignore_timestamps=ignore_timestamps)
                viz_body = [item.ast_node for item in sorted(viz_slice, key=lambda x: x.start_time)]
                new_body = astlib.prepare_body(viz_body)
                candidate = astlib.update_stmt_body(code_ast, new_body)

                #  Gather the pandas functions used
                pandas_functions_used: Set[str] = set()
                other_functions_used: Set[str] = set()
                for c_n in astlib.walk(candidate):
                    if isinstance(c_n, astlib.Call) and c_n in func_mod_name_map:

                        qual_name = func_mod_name_map[c_n] + '.' + astlib.to_code(c_n.func).split('.')[-1]
                        if func_mod_name_map[c_n].startswith("pandas"):
                            pandas_functions_used.add(qual_name)
                        elif not func_mod_name_map[c_n].startswith("builtins"):
                            other_functions_used.add(qual_name)

                df_vars: Dict[int, List[astlib.Name]] = collections.defaultdict(list)
                earliest_access: Dict[int, int] = {}
                for d in trace.get_external_dependencies_for_items(viz_slice):
                    if d.type == DependencyType.DEF_ACCESS:
                        assert isinstance(d.dst, AccessEvent)
                        if d.dst.owner is not None and d.dst.owner.ast_node in df_exprs:
                            assert isinstance(d.dst.owner.ast_node, astlib.Name)
                            df_vars[d.dst.obj_id].append(d.dst.owner.ast_node)
                            if d.dst.obj_id not in earliest_access:
                                earliest_access[d.dst.obj_id] = d.dst.timestamp
                            else:
                                earliest_access[d.dst.obj_id] = min(earliest_access[d.dst.obj_id], d.dst.timestamp)

                #  The mapping df_args will represent the arguments to be provided to the function
                #  to recreate the transformation.
                if len(df_vars) == 1:
                    var_name = "df"
                    obj_id, var_names = next(iter(df_vars.items()))
                    replacements = {v: astlib.create_name_expr(var_name) for v in var_names}
                    df_args = {var_name: df_collector.get_df_snapshot(obj_id, latest_before=earliest_access[obj_id])}
                else:
                    var_prefix = "df"
                    replacements = {}
                    df_args = {}
                    for idx, (obj_id, var_names) in enumerate(df_vars.items(), 1):
                        var_name = f"{var_prefix}{idx}"
                        replacements.update({v: astlib.create_name_expr(var_name) for v in var_names})
                        df_args[var_name] = df_collector.get_df_snapshot(obj_id, latest_before=earliest_access[obj_id])

                if len(df_args) > 2:
                    #  Ignore overly complex transformations for now.
                    continue

                #  With the arguments figured out, we can construct the desired function by creating a new function
                #  with the required signature, and making the slice the body of the function.
                candidate = astlib.with_deep_replacements(candidate, replacements)
                func_def = astlib.parse_stmt(f"def transform({', '.join(df_args.keys())}):\n    pass")
                func_def = astlib.update_stmt_body(func_def, candidate.body)

                #  If the snippet type is a df assignment or an expression to be printed, convert the last expression
                #  to a return statement.
                if snippet_type == "DF_ASSIGN" or snippet_type == "PRINT_EXPR":
                    last_stmt = list(astlib.iter_body_stmts(candidate))[-1]
                    if not isinstance(last_stmt, astlib.Expr):
                        #  This needs to just be an expression, otherwise we've made a mistake somewhere.
                        continue

                    func_def = astlib.with_deep_replacements(func_def, {
                        last_stmt: astlib.create_return(last_stmt.value)
                    })

                code = astlib.to_code(func_def)
                logger.info(f"Extracted Transformation\n{code}")

                #  We lift the hard-coded column references to column parameters.
                code_ast = astlib.parse(code)
                logger.info("Extracting Column Parameters")
                code, col_args = cls._create_col_parameters(code_ast, df_args, col_collector=col_collector)
                logger.info(f"Finished Extracting Column Parameters:\nCode:\n{code}\nArguments:\n{col_args}")

                #  We are all set. Save the output to an appropriate place.
                logger.info(f"Final Processed Visualization Function ({snippet_type}):\n{code}")

                #  Figure out the paths to store the input dataframes at.
                for df in df_args.values():
                    if id(df) not in df_obj_id_to_pkl_paths:
                        df_obj_id_to_pkl_paths[id(df)] = f"df_{len(df_obj_id_to_pkl_paths) + 1}.pkl"
                        df_obj_id_to_df[id(df)] = df

                found.append({
                    "snippet_type": snippet_type,
                    "code": code,
                    "pandas_functions": list(pandas_functions_used),
                    "other_functions": list(other_functions_used),
                    "df_args": {arg: df_obj_id_to_pkl_paths[id(df)] for arg, df in df_args.items()},
                    "col_args": col_args,
                })
            except Exception as e:
                logger.exception(e)
                continue

        logger.info(f"Found {len(found)} snippets")
        logger.info("Dumping snippets")

        mining_output_dir = os.path.join(output_dir_path, cls.__name__)
        os.makedirs(mining_output_dir, exist_ok=True)

        def str_presenter(dumper, data):
            if len(data.splitlines()) > 1:  # check for multiline string
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_presenter)

        yaml_output = [{
            "code": c
        } for c in found]
        with open(os.path.join(mining_output_dir, "pandas_functions.yaml"), "w") as f:
            yaml.dump(sorted(yaml_output, key=lambda c: len(c['code'])), f)

        logger.info("Dumping dataframes")
        for obj_id, pkl_path in df_obj_id_to_pkl_paths.items():
            df = df_obj_id_to_df[obj_id]
            df.to_pickle(os.path.join(mining_output_dir, pkl_path))

        logger.info("Finished Extraction")

    @classmethod
    def _has_non_pandas_calls(cls, node: astlib.AstNode, func_mod_name_map: Dict[astlib.Call, str]) -> bool:
        for n in astlib.walk(node):
            if isinstance(n, astlib.Call) and n in func_mod_name_map:
                qual_name = func_mod_name_map[n]
                if not qual_name.startswith("pandas."):
                    return True
                # Also we don't consider the plotting functions to be part of core pandas.
                elif qual_name.startswith("pandas.plotting"):
                    return True
                else:
                    print(qual_name)

        return False

    @classmethod
    def _get_slice(cls,
                   trace: HierarchicalTrace,
                   criteria: Set[TraceItem],
                   body_stmts: Set[astlib.AstNode],
                   ignore_timestamps: Dict[int, int]) -> List[TraceItem]:
        worklist = collections.deque(criteria)
        queued: Set[TraceItem] = set(criteria)
        disallowed: Set[TraceItem] = set()

        while len(worklist) > 0:
            item = worklist.popleft()
            for d in trace.get_external_dependencies(item):
                for i in trace.get_explicitly_resolving_items(d):
                    #  We only want to use the statements specified in body_stmts
                    if i.ast_node in body_stmts:
                        if i not in queued:
                            if d.dst.obj_id in ignore_timestamps and ignore_timestamps[d.dst.obj_id] >= i.end_time:
                                disallowed.add(i)
                            queued.add(i)
                            worklist.append(i)
                            break

        worklist = collections.deque(criteria)
        queued: Set[TraceItem] = set(criteria)

        while len(worklist) > 0:
            item = worklist.popleft()
            for d in trace.get_external_dependencies(item):
                for i in trace.get_explicitly_resolving_items(d):
                    #  We only want to use the statements specified in body_stmts
                    if i.ast_node in body_stmts:
                        if (i not in queued) and (i not in disallowed):
                            queued.add(i)
                            worklist.append(i)
                            break

        return sorted(queued, key=lambda x: x.start_time)

    @classmethod
    def _collect_col_attribute_access(cls, code_ast: astlib.AstNode, pos_args: List[Any], kw_args: Dict[str, Any]
                                      ) -> Dict[astlib.Attribute, str]:
        attr_access_collector = DfColAttrAccessCollector()
        instrumenter = Instrumenter(Instrumentation.from_generators(attr_access_collector))

        #  Run the instrumenter, and execute.
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)
        globs['transform'](*pos_args, **kw_args)

        return attr_access_collector.get_df_col_attr_accesses()

    @classmethod
    def _create_col_parameters(cls,
                               code_ast: astlib.AstNode,
                               df_args: Dict[str, pd.DataFrame],
                               col_collector: DfStrColumnsCollector) -> Tuple[str, Dict[str, str]]:

        #  We will differentiate between columns that are present in the input dataframes, versus those
        #  that are created in the visualization
        inp_df_cols = set()
        for df in df_args.values():
            if df.columns.nlevels == 1:
                inp_df_cols.update(c for c in df.columns if isinstance(c, str))
            else:
                flattened = sum((list(i) for i in df.columns.levels), [])
                inp_df_cols.update(c for c in flattened if isinstance(c, str))

        other_df_cols = col_collector.get_collected_cols() - inp_df_cols
        all_cols = inp_df_cols | other_df_cols

        #  Both df["col_name"]  and df.col_name are equivalent ways of referring to a dataframe column.
        #  Identifying strings is easy, but to know if the column is being referred to using an attribute access,
        #  we need to employ some more instrumentation. So we do that to get all the attribute accesses
        #  that correspond to a column access of a dataframe.
        attr_accesses = cls._collect_col_attribute_access(code_ast,
                                                          pos_args=[],
                                                          kw_args={arg: df.copy() for arg, df in df_args.items()})

        #  All set to extract columns
        num_inp_df_cols_found = 0
        num_other_df_cols_found = 0
        cols_to_params: Dict[str, str] = {}
        node_replacement_map: Dict[astlib.AstNode, astlib.AstNode] = {}
        column_list_exprs: Dict[str, Set[Optional[astlib.List]]] = collections.defaultdict(set)
        column_list_cands: Set[astlib.List] = set()
        for node in astlib.walk(code_ast):
            if isinstance(node, astlib.SimpleString):
                #  We apply the heuristic that if this matches a known column name, it's safe to lift it up to
                #  a parameter. Of course if the strings are not intended to be columns, this will return a wrong
                #  result, but that isn't very likely so we'll take the hit.
                if node.evaluated_value in all_cols:
                    #  Okay it matches a known column
                    col_name = node.evaluated_value
                    if col_name not in cols_to_params:
                        if col_name in inp_df_cols:
                            num_inp_df_cols_found += 1
                            new_param = f"col{num_inp_df_cols_found}"
                        else:
                            num_other_df_cols_found += 1
                            new_param = f"new_col{num_other_df_cols_found}"

                        cols_to_params[col_name] = new_param
                        print("NEW", cols_to_params)

                    node_replacement_map[node] = astlib.create_name_expr(cols_to_params[col_name])
                    parent = astlib.get_parent(astlib.get_parent(node, code_ast), code_ast)
                    if isinstance(parent, astlib.List) and len(parent.elements) > 1:
                        if all(isinstance(elem.value, astlib.SimpleString) and elem.value.evaluated_value in inp_df_cols
                               for elem in parent.elements):
                            column_list_exprs[col_name].add(parent)
                            column_list_cands.add(parent)
                        else:
                            column_list_exprs[col_name].add(None)
                    else:
                        column_list_exprs[col_name].add(None)

            elif isinstance(node, astlib.Attribute) and node in attr_accesses:
                #  It's an attribute-based column access. We'll replace it with a subscript access i.e. (df["col_name"])
                col_name = attr_accesses[node]
                if col_name not in cols_to_params:
                    if col_name in inp_df_cols:
                        num_inp_df_cols_found += 1
                        new_param = f"col{num_inp_df_cols_found}"
                    else:
                        num_other_df_cols_found += 1
                        new_param = f"new_col{num_other_df_cols_found}"

                    cols_to_params[col_name] = new_param
                    column_list_exprs[col_name].add(None)

                node_replacement_map[node] = astlib.create_subscript_expr(node.value, [cols_to_params[col_name]])

        for col_name, col_list_exprs in column_list_exprs.items():
            valid = True
            if None in col_list_exprs:
                valid = False
            elif len(col_list_exprs) > 1:
                if len({astlib.to_code(expr) for expr in col_list_exprs}) > 1:
                    valid = False

            if not valid:
                for expr in col_list_exprs:
                    if expr is not None:
                        column_list_cands.discard(expr)

        if len(column_list_cands) > 0:
            eq_list_map: Dict[str, List[astlib.List]] = collections.defaultdict(list)
            for cand in column_list_cands:
                eq_list_map[astlib.to_code(cand)].append(cand)
                for elem in cand.elements:
                    if elem.value.evaluated_value in cols_to_params:
                        cols_to_params.pop(elem.value.evaluated_value)

            if len(eq_list_map) == 1:
                cols_to_params[next(iter(eq_list_map.keys()))] = "columns"
            else:
                for idx, val in enumerate(eq_list_map.keys(), 1):
                    cols_to_params[val] = f"columns{idx}"

            for rep, cands in eq_list_map.items():
                var_name = cols_to_params[rep]
                for cand in cands:
                    node_replacement_map[cand] = astlib.create_name_expr(var_name)

        #  Replace the strings and attribute access with the new name accesses and subscript accesses respectively.
        updated_code_ast = astlib.with_deep_replacements(code_ast, node_replacement_map)
        if isinstance(updated_code_ast, astlib.Module):
            func_def = next(astlib.iter_body_stmts(updated_code_ast))
        elif isinstance(updated_code_ast, astlib.FunctionDef):
            func_def = updated_code_ast
        else:
            assert False

        #  Also need to update the signature to reflect the new parameters.
        new_sig: str = ", ".join(sorted(df_args.keys()) + sorted(cols_to_params.values()))
        updated_func_def = astlib.parse_stmt(f"def transform({new_sig}):\n    pass")
        updated_func_def = astlib.update_stmt_body(updated_func_def,
                                                   list(astlib.prepare_body(list(astlib.iter_body_stmts(func_def)))))

        new_code = astlib.to_code(updated_func_def)
        col_arguments = {v: eval(k) if k.startswith("[") else k for k, v in cols_to_params.items()}

        return new_code, col_arguments
