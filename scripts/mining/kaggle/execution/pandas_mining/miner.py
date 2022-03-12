import collections
import os
import sys
from typing import Set, Dict, List, Tuple, Iterator, Any

import attrs
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from databutler.datana.viz.utils import mpl_exec
from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, ObjWriteEvent, TraceItem
from databutler.pat.analysis.instrumentation import ExprCallbacksGenerator, ExprCallback, StmtCallbacksGenerator, \
    StmtCallback, Instrumentation, Instrumenter, ExprWrappersGenerator, ExprWrapper
from databutler.utils import inspection
from databutler.utils.logging import logger
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.execution.instrumentation_utils import IPythonMagicBlocker
from scripts.mining.kaggle.execution.mpl_seaborn_mining.minimization import minimize_code
from scripts.mining.kaggle.execution.mpl_seaborn_mining.var_optimization import optimize_vars
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType


@attrs.define(eq=False, repr=False)
class FuncNameFinder(ExprWrappersGenerator):
    _func_calls_to_name: Dict[astlib.Call, str] = attrs.field(init=False, factory=dict)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                yield expr.func, ExprWrapper(
                    callable=self._gen_func_name_finding_wrapper(expr),
                    name=self.gen_wrapper_id(),
                )

    def _gen_func_name_finding_wrapper(self, call_expr: astlib.Call):
        def wrapper(value):
            qual_name = inspection.get_fully_qualified_name(value)
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
        func_name_finder = FuncNameFinder()
        #  Ready up the instrumentation for the df detectors.
        df_collector = ReadCsvDfCollector()
        col_collector = DfStrColumnsCollector()
        #  Need to avoid executing matplotlib magics as they can mess with the config.
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        name_finding_instrumentation = Instrumentation.from_generators(func_name_finder)
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
                                 func_name_finder=func_name_finder,
                                 df_collector=df_collector, col_collector=col_collector,
                                 output_dir_path=output_dir_path)

    @classmethod
    def _extract_pandas_code(cls, code_ast: astlib.AstNode,
                             trace: HierarchicalTrace,
                             func_name_finder: FuncNameFinder,
                             df_collector: ReadCsvDfCollector,
                             col_collector: DfStrColumnsCollector,
                             output_dir_path: str):

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
        worklist = collections.deque(fwd_slicing_items)

        while len(worklist) > 0:
            item = worklist.popleft()

            found = False
            for dep in trace.get_forward_dependencies(item):
                for dep_item in trace.get_affording_items(dep.dst):
                    if dep_item.ast_node in body_stmts:
                        found = True
                        worklist.append(dep_item)
                        break

            if not found:
                fwd_slicing_leaves.add(item)

        print(f"FOUND {len(fwd_slicing_leaves)} leaves")
        for leaf in fwd_slicing_leaves:
            print("FOUND", astlib.to_code(leaf.ast_node))



