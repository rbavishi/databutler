import collections
import os
from typing import Set, Dict, List, Tuple, Iterator, Deque

import attrs
import pandas as pd
import yaml

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, TraceItem
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
        #  Need to avoid executing matplotlib magics as they can mess with the config.
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        name_finding_instrumentation = Instrumentation.from_generators(func_mod_name_finder)
        df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        col_collector_instrumentation = Instrumentation.from_generators(col_collector)
        magic_blocker_instrumentation = Instrumentation.from_generators(magic_blocker)

        #  Merge all the instrumentation together.
        instrumentation = (trace_instrumentation |
                           name_finding_instrumentation |
                           df_collector_instrumentation |
                           col_collector_instrumentation |
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
        exec(new_code, globs, globs)

        #  Since we added the tracing instrumentation, we will now we able to extract the trace of the program.
        #  This trace contains all the information we need to extract dependencies, and do things like slicing.
        trace = trace_instrumentation.get_hierarchical_trace()

        #  Use the trace, matplotlib figure and df detectors to extract visualization code.
        cls._extract_pandas_code(code_ast, trace,
                                 func_mod_name_finder=func_mod_name_finder,
                                 df_collector=df_collector, col_collector=col_collector,
                                 output_dir_path=output_dir_path)

    @classmethod
    def _extract_pandas_code(cls, code_ast: astlib.AstNode,
                             trace: HierarchicalTrace,
                             func_mod_name_finder: FuncModNameFinder,
                             df_collector: ReadCsvDfCollector,
                             col_collector: DfStrColumnsCollector,
                             output_dir_path: str):

        func_mod_name_map: Dict[astlib.Call, str] = func_mod_name_finder.get_func_calls_to_names()

        #  We want to use the top-level statements of the notebook in the extracted code, so we compute and keep aside.
        body_stmts = set()
        for s in astlib.iter_body_stmts(code_ast):
            if isinstance(s, astlib.NotebookCell):
                body_stmts.update(astlib.iter_body_stmts(s.body))
            else:
                body_stmts.add(s)

        fwd_slicing_nodes: Set[astlib.AstNode] = set()

        for read_csv_expr, df in df_collector.get_collected_dfs().items():
            for parent in astlib.iter_parents(read_csv_expr, context=code_ast):
                if parent in body_stmts:
                    fwd_slicing_nodes.add(parent)
                    break

        fwd_slicing_items: List[TraceItem] = []
        for item in trace.items:
            if item.ast_node in fwd_slicing_nodes:
                fwd_slicing_items.append(item)

        fwd_slicing_leaves: Set[TraceItem] = set()
        worklist: Deque[TraceItem] = collections.deque(fwd_slicing_items)
        seen: Set[TraceItem] = set()

        logger.info("Finding Criteria for Backward Slicing using Forward Slicing")

        while len(worklist) > 0:
            item = worklist.popleft()
            if item in seen:
                continue

            seen.add(item)

            found = False
            for dep in trace.get_forward_dependencies(item):
                for dep_item in trace.get_affording_items(dep.dst):
                    if dep_item.ast_node in body_stmts:
                        #  Confirm there are no non-pandas calls.
                        if cls._has_non_pandas_calls(dep_item.ast_node, func_mod_name_map):
                            continue

                        found = True
                        worklist.append(dep_item)
                        break

            if (not found) and item not in fwd_slicing_items:
                fwd_slicing_leaves.add(item)

        logger.info(f"Found {len(fwd_slicing_leaves)} leaves")

        #  Perform backward slicing from these leaves.
        logger.info("Starting Backward Slicing")
        raw_slices: Set[str] = set()
        for criterion in fwd_slicing_leaves:
            pandas_slice: List[TraceItem] = cls._get_slice(trace, {criterion}, body_stmts)
            body = [item.ast_node for item in sorted(pandas_slice, key=lambda x: x.start_time)]
            new_body = astlib.prepare_body(body)
            candidate = astlib.update_stmt_body(code_ast, new_body)
            code = astlib.to_code(candidate)

            logger.info(f"Extracted Raw Visualization Slice:\n{code}")
            raw_slices.add(code)

        logger.info(f"Found {len(raw_slices)} raw slices")

        def str_presenter(dumper, data):
            if len(data.splitlines()) > 1:  # check for multiline string
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_presenter)

        logger.info("Dumping slices")

        mining_output_dir = os.path.join(output_dir_path, cls.__name__)
        os.makedirs(mining_output_dir, exist_ok=True)

        with open(os.path.join(mining_output_dir, "viz_functions.yaml"), "w") as f:
            yaml.dump(sorted(raw_slices, key=len), f)

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
                   body_stmts: Set[astlib.AstNode]) -> List[TraceItem]:
        worklist = collections.deque(criteria)
        queued: Set[TraceItem] = set(criteria)

        while len(worklist) > 0:
            item = worklist.popleft()
            for d in trace.get_external_dependencies(item):
                for i in trace.get_explicitly_resolving_items(d):
                    #  We only want to use the statements specified in body_stmts
                    if i.ast_node in body_stmts:
                        if i not in queued:
                            queued.add(i)
                            worklist.append(i)
                            break

        return sorted(queued, key=lambda x: x.start_time)

