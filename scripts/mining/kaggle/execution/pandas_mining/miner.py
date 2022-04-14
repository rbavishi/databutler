import collections
import os
from typing import Set, Dict, List, Tuple, Iterator, Deque

import attrs
import pandas as pd
import yaml

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, TraceItem, ObjWriteEvent, DefEvent, \
    AccessEvent
from databutler.pat.analysis.instrumentation import Instrumentation, Instrumenter, ExprWrappersGenerator, ExprWrapper
from databutler.utils import inspection
from databutler.utils.logging import logger
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.execution.instrumentation_utils import IPythonMagicBlocker
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType


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
class ReadCsvDfCollector(ExprWrappersGenerator):
    #  Internal
    _collected_dfs: Dict[astlib.Call, pd.DataFrame] = attrs.field(init=False, factory=dict)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call) and "read_csv" in astlib.to_code(expr.func):
                yield expr, ExprWrapper(
                    callable=self._gen_collecting_wrapper(expr),
                    name=self.gen_wrapper_id(),
                )

    def _gen_collecting_wrapper(self, expr: astlib.Call):
        def wrapper(value):
            if isinstance(value, pd.DataFrame):
                self._collected_dfs[expr] = value.copy()

            return value

        return wrapper

    def get_collected_dfs(self) -> Dict[astlib.Call, pd.DataFrame]:
        return self._collected_dfs.copy()


@attrs.define(eq=False, repr=False)
class DfObjIdCollector(ExprWrappersGenerator):
    _df_obj_ids: Set[int] = attrs.field(init=False, factory=set)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            yield expr, ExprWrapper(
                callable=self._collect_df_obj_ids,
                name=self.gen_wrapper_id(),
            )

    def _collect_df_obj_ids(self, value):
        if isinstance(value, pd.DataFrame):
            self._df_obj_ids.add(id(value))

        return value

    def get_df_obj_ids(self) -> Set[int]:
        return self._df_obj_ids.copy()


@attrs.define(eq=False, repr=False)
class DfExprCollector(ExprWrappersGenerator):
    _exprs: Set[astlib.BaseExpression] = attrs.field(init=False, factory=set)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            yield expr, ExprWrapper(
                callable=self._gen_collect_expr(expr),
                name=self.gen_wrapper_id(),
            )

    def _gen_collect_expr(self, expr):
        def wrapper(value):
            if isinstance(value, pd.DataFrame):
                self._exprs.add(expr)

            return value

        return wrapper

    def get_df_exprs(self) -> Set[astlib.BaseExpression]:
        return self._exprs.copy()


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
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Attribute):
                yield expr.value, ExprWrapper(
                    callable=self._gen_df_detector(expr.value),
                    name=self.gen_wrapper_id(),
                )
                yield expr, ExprWrapper(
                    callable=self._gen_collector(expr),
                    name=self.gen_wrapper_id(),
                )

    def _gen_df_detector(self, expr: astlib.BaseExpression):
        def wrapper(value):
            if isinstance(value, pd.DataFrame):
                self._df_exprs.add(expr)

            return value

        return wrapper

    def _gen_collector(self, expr: astlib.Attribute):
        attr_name = expr.attr.value

        def wrapper(value):
            if expr.value in self._df_exprs and isinstance(value, pd.Series):
                self._collected_accesses[expr] = attr_name

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
        df_collector = ReadCsvDfCollector()
        col_collector = DfStrColumnsCollector()
        #  Collect object ids of the dataframes
        df_obj_id_collector = DfObjIdCollector()
        #  Collect exprs that evaluate to dataframes
        df_expr_collector = DfExprCollector()
        #  Collect last stmts of notebook cells that evaluate to something that is printed
        nb_print_expr_collector = NotebookPrintCollector()
        #  Need to avoid executing matplotlib magics as they can mess with the config.
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        name_finding_instrumentation = Instrumentation.from_generators(func_mod_name_finder)
        df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        col_collector_instrumentation = Instrumentation.from_generators(col_collector)
        magic_blocker_instrumentation = Instrumentation.from_generators(magic_blocker)
        df_obj_id_instrumentation = Instrumentation.from_generators(df_obj_id_collector)
        df_expr_collector_instrumentation = Instrumentation.from_generators(df_expr_collector)
        nb_print_expr_collector_instrumentation = Instrumentation.from_generators(nb_print_expr_collector)

        #  Merge all the instrumentation together.
        instrumentation = (trace_instrumentation |
                           name_finding_instrumentation |
                           df_collector_instrumentation |
                           col_collector_instrumentation |
                           df_obj_id_instrumentation |
                           df_expr_collector_instrumentation |
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
                                 df_obj_id_collector=df_obj_id_collector,
                                 df_expr_collector=df_expr_collector,
                                 nb_print_expr_collector=nb_print_expr_collector,
                                 output_dir_path=output_dir_path)

    @classmethod
    def _extract_pandas_code(cls, code_ast: astlib.AstNode,
                             trace: HierarchicalTrace,
                             func_mod_name_finder: FuncModNameFinder,
                             df_collector: ReadCsvDfCollector,
                             col_collector: DfStrColumnsCollector,
                             df_obj_id_collector: DfObjIdCollector,
                             df_expr_collector: DfExprCollector,
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
        for read_csv_expr, df in df_collector.get_collected_dfs().items():
            for parent in astlib.iter_parents(read_csv_expr, context=code_ast):
                if parent in body_stmts:
                    allowed_slicing_stmts.discard(parent)
                    break

        df_obj_ids: Set[int] = df_obj_id_collector.get_df_obj_ids()
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

        for read_csv_expr, df in df_collector.get_collected_dfs().items():
            for parent in astlib.iter_parents(read_csv_expr, context=code_ast):
                if parent in body_stmts:
                    disallowed_dependencies.add(parent)

        criteria_items = set()
        for df_writing_items in obj_id_to_writing_items.values():
            criteria_items.update((item, "DF_WRITE") for item in df_writing_items)

        df_exprs = df_expr_collector.get_df_exprs()
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

        for item, snippet_type in criteria_items:
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
            for c_n in astlib.walk(candidate):
                if isinstance(c_n, astlib.Call) and c_n in func_mod_name_map and \
                        func_mod_name_map[c_n].startswith("pandas"):
                    pandas_functions_used.add(func_mod_name_map[c_n] + '.' + astlib.to_code(c_n.func).split('.')[-1])

            found.append({
                "snippet_type": snippet_type,
                "code": astlib.to_code(candidate),
                "pandas_functions": list(pandas_functions_used),
            })

            print("-----")
            print(snippet_type)
            print(found[-1]['code'])
            print("-----")
            print(found[-1]['pandas_functions'])

        print("-----")

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
