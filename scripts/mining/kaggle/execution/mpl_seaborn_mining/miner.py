import collections
from typing import Set, Dict, List, Tuple, Iterator

import attrs
import pandas as pd

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, ObjWriteEvent, TraceItem
from databutler.pat.analysis.instrumentation import ExprCallbacksGenerator, ExprCallback, StmtCallbacksGenerator, \
    StmtCallback, Instrumentation, Instrumenter, ExprWrappersGenerator, ExprWrapper
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.execution.instrumentation_utils import IPythonMagicBlocker

from matplotlib import pyplot as plt

from scripts.mining.kaggle.execution.mpl_seaborn_mining.var_optimization import optimize_vars
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType


@attrs.define(eq=False, repr=False)
class MatplotlibFigureDetector(ExprCallbacksGenerator):
    """
    Capture all the unique matplotlib figure objects observed during the execution of the program.
    At any given point, `plt.gcf()` returns the current active figure object. Therefore, after every
    call expression, we add the result of `plt.gcf()` to a set. This makes sense as all interactions with
    matplotlib mostly happen through function calls. We do this by using expression callbacks that are
    called after every call expression in the AST is evaluated.
    """

    #  Internal
    _found_objects: Set[plt.Figure] = attrs.field(init=False, factory=set)

    def gen_expr_callbacks_simple(self, ast_root: astlib.AstNode
                                  ) -> Iterator[Tuple[astlib.BaseExpression, ExprCallback]]:
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                yield expr, ExprCallback(
                    callable=lambda: self._found_objects.add(plt.gcf()),
                    name=self.gen_expr_callback_id(),
                    position='post',
                    arg_str=''
                )

    def get_matplotlib_figure_objects(self) -> Set[plt.Figure]:
        return self._found_objects


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
                self._collected_dfs[expr] = value

            return value

        return wrapper

    def get_collected_dfs(self) -> Dict[astlib.Call, pd.DataFrame]:
        return self._collected_dfs.copy()


@attrs.define(eq=False, repr=False)
class NotebookExecutionManager(StmtCallbacksGenerator):
    """
    We execute notebooks by concatenating all cells into a single python script/module.
    This creates a problem for visualizations, because each cell by default is associated with a fresh
    figure object. The purpose of this instrumentation is to add a statement callback after
    every NotebookCell node, that explicitly creates a figure object using `plt.figure()`.
    Note that our AST library has first-class support for Notebook cells.
    """

    def gen_stmt_callbacks_simple(self, ast_root: astlib.AstNode
                                  ) -> Iterator[Tuple[astlib.AstStatementT, StmtCallback]]:
        for n in self.iter_stmts(ast_root):
            if isinstance(n, astlib.NotebookCell) and len(n.body.body) > 0:
                yield n, StmtCallback(callable=lambda: plt.figure(),
                                      name=self.gen_stmt_callback_id(),
                                      position='post',
                                      arg_str='',
                                      mandatory=True)


@attrs.define(eq=False, repr=False)
class MplSeabornVizMiner(BaseExecutor):
    @classmethod
    @register_runner(name="mpl_seaborn_viz_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        #  A clock is critical in identifying dependencies.
        clock = LogicalClock()
        #  Trace instrumentation does the heavy-lifting of recording reads/writes, var. defs and their uses.
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        #  Ready up the instrumentation for the matplotlib and df detectors.
        matplotlib_detector = MatplotlibFigureDetector()
        df_collector = ReadCsvDfCollector()
        #  Need to avoid executing matplotlib magics as they can mess with the config.
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        matplotlib_instrumentation = Instrumentation.from_generators(matplotlib_detector)
        df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        magic_blocker_instrumentation = Instrumentation.from_generators(magic_blocker)
        #  Also include the trick for handling the notebook execution quirk for visualizations.
        notebook_manager_instrumentation = Instrumentation.from_generators(NotebookExecutionManager())

        #  Merge all the instrumentation together.
        instrumentation = (trace_instrumentation | df_collector_instrumentation |
                           matplotlib_instrumentation | notebook_manager_instrumentation |
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
        cls._extract_viz_code(code_ast, trace, df_collector=df_collector, fig_detector=matplotlib_detector)

    @classmethod
    def _extract_viz_code(cls, code_ast: astlib.AstNode, trace: HierarchicalTrace,
                          df_collector: ReadCsvDfCollector,
                          fig_detector: MatplotlibFigureDetector):
        #  Get all the matplotlib objects seen during the course of execution of the notebook.
        mpl_objs = fig_detector.get_matplotlib_figure_objects()
        #  The hierarchical trace uses object-IDs to identify objects instead of directly storing them.
        #  So we create a map from figure object IDs to the figures themselves for convenience.
        obj_id_to_fig: Dict[int, plt.Figure] = {id(o): o for o in mpl_objs}

        #  We want to use the top-level statements of the notebook in the visualization, so we compute and keep aside.
        #  See _get_slice to get a sense of how these statements are being used.
        body_stmts = set()
        for s in astlib.iter_body_stmts(code_ast):
            if isinstance(s, astlib.NotebookCell):
                body_stmts.update(astlib.iter_body_stmts(s.body))
            else:
                body_stmts.add(s)

        #  For each figure, we collect the body statements (trace items actually) that directly write to the figure.
        #  This set of statements will constitute the slicing criterion.
        fig_to_slicing_criteria: Dict[int, Set[TraceItem]] = collections.defaultdict(set)
        for e in trace.get_events():
            if isinstance(e, ObjWriteEvent) and e.obj_id in obj_id_to_fig:
                item = e.owner
                if item.ast_node in body_stmts:
                    fig_to_slicing_criteria[e.obj_id].add(item)
                else:
                    for i in trace.iter_parents(item):
                        if i.ast_node in body_stmts:
                            fig_to_slicing_criteria[e.obj_id].add(i)
                            break

        #  For each non-empty figure, we'll perform slicing using the collected criteria,
        print("STARTING")
        for obj_id, fig in obj_id_to_fig.items():
            if obj_id not in fig_to_slicing_criteria:
                if len(fig.axes) != 0:
                    print("ERROR?")
                #  There's no write to this figure, ignore.
                continue

            if len(fig.axes) == 0:
                #  Need to have at least one axis to be non-empty.
                continue

            print("FOUND FIGURE")
            #  Extract the slice as a list of trace items, whose corresponding ast node are the ones we will
            #  use to build up the body of our visualization.
            criteria = fig_to_slicing_criteria[obj_id]
            viz_slice: List[TraceItem] = cls._get_slice(trace, criteria, body_stmts)
            viz_body = [item.ast_node for item in sorted(viz_slice, key=lambda x: x.start_time)]
            new_body = astlib.prepare_body(viz_body)
            candidate = astlib.update_stmt_body(code_ast, new_body)

            all_nodes = set(astlib.walk(candidate))
            to_replace: List[Tuple[astlib.AstNode, pd.DataFrame]] = []
            for node, df in df_collector.get_collected_dfs().items():
                if node in all_nodes:
                    to_replace.append((node, df))

            if len(to_replace) == 1:
                var_name = "_df"
                replacements = {to_replace[0][0]: astlib.create_name_expr(var_name)}
                df_args = {var_name: to_replace[0][1]}
            else:
                var_prefix = "_df_"
                replacements = {node: astlib.create_name_expr(f"{var_prefix}{idx}")
                                for idx, (node, df) in enumerate(to_replace, 1)}
                df_args = {f"{var_prefix}{idx}": df for idx, (node, df) in enumerate(to_replace, 1)}

            candidate = astlib.with_deep_replacements(candidate, replacements)
            func_def = astlib.parse_stmt(f"def visualization({', '.join(i.value for i in replacements.values())}):\n    pass")
            func_def = astlib.update_stmt_body(func_def, candidate.body)
            code = astlib.to_code(func_def)
            print(code)
            print("\n---\nOPTIMIZED\n---")
            print(optimize_vars(code, [], df_args))

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
