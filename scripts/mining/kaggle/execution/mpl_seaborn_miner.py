import collections
from typing import Set, Dict, List

import attrs

from databutler.pat import astlib
from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder import get_hierarchical_trace_instrumentation
from databutler.pat.analysis.hierarchical_trace.core import HierarchicalTrace, ObjWriteEvent, TraceItem
from databutler.pat.analysis.instrumentation import ExprCallbacksGenerator, ExprCallback, StmtCallbacksGenerator, \
    StmtCallback, Instrumentation, Instrumenter
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.execution.instrumentation_utils import IPythonMagicBlocker

from matplotlib import pyplot as plt

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

    def reset(self):
        self._found_objects.clear()

    def gen_expr_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
        callbacks: Dict[astlib.BaseExpression, List[ExprCallback]] = collections.defaultdict(list)

        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                callbacks[expr].append(self.get_callback(expr))

        return callbacks

    def get_callback(self, call_expr: astlib.Call) -> ExprCallback:
        def callback():
            self._found_objects.add(plt.gcf())

        return ExprCallback(callable=callback,
                            name=self.gen_expr_callback_id(),
                            position='post',
                            arg_str='')

    def get_matplotlib_figure_objects(self) -> Set[plt.Figure]:
        """
        :return:
        """
        return self._found_objects


@attrs.define(eq=False, repr=False)
class NotebookExecutionManager(StmtCallbacksGenerator):
    """
    We execute notebooks by concatenating all cells into a single python script/module.
    This creates a problem for visualizations, because each cell by default is associated with a fresh
    figure object. The purpose of this instrumentation is to add a statement callback after
    every NotebookCell node, that explicitly creates a figure object using `plt.figure()`.
    Note that our AST library has first-class support for Notebook cells.
    """

    def gen_stmt_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.AstStatementT,
                                                                   List[StmtCallback]]:
        callbacks: Dict[astlib.AstStatementT, List[StmtCallback]] = collections.defaultdict(list)
        for n in self.iter_stmts(ast_root):
            if isinstance(n, astlib.NotebookCell) and len(n.body.body) > 0:
                callbacks[n].append(StmtCallback(callable=self.post_cell_callback,
                                                 name=self.gen_stmt_callback_id(),
                                                 position='post',
                                                 arg_str='',
                                                 mandatory=True))

        return callbacks

    def post_cell_callback(self):
        plt.figure()


@attrs.define(eq=False, repr=False)
class MplSeabornVizMiner(BaseExecutor):
    @classmethod
    @register_runner(name="mpl_seaborn_viz_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        clock = LogicalClock()
        #  Trace instrumentation does the heavy-lifting of recording reads/writes, var. defs and their uses.
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        #  Ready up the instrumentation for the matplotlib and df detectors.
        matplotlib_detector = MatplotlibFigureDetector()
        magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})

        matplotlib_instrumentation = Instrumentation.from_generators(matplotlib_detector)
        magic_blocker_instrumentation = Instrumentation.from_generators(magic_blocker)
        #  Also include the trick for handling the notebook execution quirk for visualizations.
        notebook_manager_instrumentation = Instrumentation.from_generators(NotebookExecutionManager())
        instrumentation = (trace_instrumentation |
                           matplotlib_instrumentation | notebook_manager_instrumentation |
                           magic_blocker_instrumentation)

        instrumenter = Instrumenter(instrumentation)

        if source_type == KaggleNotebookSourceType.IPYTHON_NOTEBOOK:
            code_ast = astlib.parse(source, extension='.ipynb')
        elif source_type == KaggleNotebookSourceType.PYTHON_SOURCE_FILE:
            code_ast = astlib.parse(source)
        else:
            raise NotImplementedError(f"Could not recognize source of type {source_type}")

        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        trace = trace_instrumentation.get_hierarchical_trace()

        cls._extract_raw_slices(code_ast, trace, fig_detector=matplotlib_detector)

        print("STDOUT HELLO WORLD")
        return "Hello World"

    @classmethod
    def _extract_raw_slices(cls,
                            code_ast: astlib.AstNode,
                            trace: HierarchicalTrace,
                            fig_detector: MatplotlibFigureDetector,
                            ):

        mpl_objs = fig_detector.get_matplotlib_figure_objects()
        fig_ids = {id(o) for o in mpl_objs}
        fig_id_to_fig = {id(o): o for o in mpl_objs}

        if not isinstance(code_ast, astlib.Module):
            raise AssertionError("Ast should be a module")

        if len(code_ast.body) == 0:
            return

        body_stmts = []

        for s in astlib.iter_body_stmts(code_ast):
            if isinstance(s, astlib.NotebookCell):
                body_stmts.extend(astlib.iter_body_stmts(s.body))
            else:
                body_stmts.append(s)

        body_stmts = set(body_stmts)

        plotting_items = collections.defaultdict(set)
        for e in trace.get_events():
            if isinstance(e, ObjWriteEvent) and e.obj_id in fig_ids:
                item = e.owner
                if item.ast_node in body_stmts:
                    plotting_items[e.obj_id].add(item)
                else:
                    for i in trace.iter_parents(item):
                        if i.ast_node in body_stmts:
                            plotting_items[e.obj_id].add(i)
                            break

        for fig in fig_ids:
            if fig not in plotting_items:
                continue

            fig_obj = fig_id_to_fig[fig]
            if len(fig_obj.axes) == 0:
                continue

            criteria = plotting_items[fig]
            print("YAY", cls.extract_plotting_code(trace, criteria, body_stmts))
            print("Found figure")

            required_items = cls.extract_plotting_code(trace, criteria, body_stmts)
            new_body = [i.ast_node for i in sorted(required_items, key=lambda x: x.start_time)]
            new_body = astlib.prepare_body(new_body)
            candidate = astlib.update_stmt_body(code_ast, new_body)
            print(astlib.to_code(candidate))

    @classmethod
    def extract_plotting_code(cls,
                              trace: HierarchicalTrace,
                              plotting_items: Set[TraceItem],
                              body_stmts: Set[astlib.AstNode]):
        worklist = collections.deque(plotting_items)
        queued: Set[TraceItem] = set(plotting_items)

        while len(worklist) > 0:
            item = worklist.popleft()
            for d in trace.get_external_dependencies(item):
                for i in trace.get_explicitly_resolving_items(d):
                    if i.ast_node in body_stmts:
                        if i not in queued:
                            queued.add(i)
                            worklist.append(i)
                            break

        return sorted(queued, key=lambda x: x.start_time)
