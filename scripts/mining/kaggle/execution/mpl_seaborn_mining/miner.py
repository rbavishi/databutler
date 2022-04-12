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
from databutler.utils.logging import logger
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.execution.instrumentation_utils import IPythonMagicBlocker
from scripts.mining.kaggle.execution.mpl_seaborn_mining.minimization import minimize_code
from scripts.mining.kaggle.execution.mpl_seaborn_mining.var_optimization import optimize_vars
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType
from scripts.mining.kaggle.execution.vizminer import VizMiner

_MAX_VIZ_FUNC_EXEC_TIME = 5


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
class MplSeabornVizMiner(VizMiner):

    def __init__(self):
        super().__init__()

        self.fig_detector = MatplotlibFigureDetector()
        self.magic_blocker = IPythonMagicBlocker(to_block={'matplotlib'})
        self.notebook_manager = NotebookExecutionManager()
        self.mpl_collectors = [self.fig_detector, self.magic_blocker, self.notebook_manager]
        self.collectors.extend(self.mpl_collectors)

    @classmethod
    @register_runner(name="mpl_seaborn_viz_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        miner = MplSeabornVizMiner()
        instrumenter = miner.get_instrumenter()
        code_ast = miner.get_code_ast(source, source_type)

        #  Run the instrumenter, and execute.
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        #  Since we added the tracing instrumentation, we will now we able to extract the trace of the program.
        #  This trace contains all the information we need to extract dependencies, and do things like slicing.
        trace = miner.trace_instrumentation.get_hierarchical_trace()

        #  Use the trace, matplotlib figure and df detectors to extract visualization code.
        miner._extract_viz_code(code_ast, trace, output_dir_path=output_dir_path)

    def _extract_viz_code(self, code_ast: astlib.AstNode, trace: HierarchicalTrace, output_dir_path: str):

        body_stmts = self.get_body_stmts(code_ast)
        fig_to_slicing_criteria = self.get_fig_to_slicing_criteria(trace, body_stmts)


        #  For each non-empty figure, we'll perform slicing using the collected criteria,
        viz_code: List[Dict] = []
        df_obj_id_to_pkl_paths: Dict[int, str] = {}
        df_obj_id_to_df: Dict[int, pd.DataFrame] = {}

        #  Get all the matplotlib objects seen during the course of execution of the notebook.
        mpl_objs = self.fig_detector.get_matplotlib_figure_objects()
        #  The hierarchical trace uses object-IDs to identify objects instead of directly storing them.
        #  So we create a map from figure object IDs to the figures themselves for convenience.
        obj_id_to_fig: Dict[int, plt.Figure] = {id(o): o for o in mpl_objs}

        logger.info("Extracting Visualization Code")
        for obj_id, fig in obj_id_to_fig.items():
            if self.ignore_figure(obj_id, fig, fig_to_slicing_criteria):
                continue

            #  Extract the slice as a list of trace items, whose corresponding ast node are the ones we will
            #  use to build up the body of our visualization.
            criteria = fig_to_slicing_criteria[obj_id]
            viz_body = self.get_viz_body(criteria, trace, body_stmts)
            new_body = astlib.prepare_body(viz_body)
            candidate = astlib.update_stmt_body(code_ast, new_body)

            logger.info(f"Extracted Raw Visualization Slice:\n{astlib.to_code(candidate)}")

            # We convert the extracted visualization code to a function template
            code, df_args = self.df_template_func(candidate)
            logger.info(f"Extracted Visualization Function:\n{code}")

            #  Check if it executes correctly, and finishes under a fixed timeout
            if not self._check_execution(code, pos_args=[], kw_args=df_args, timeout=_MAX_VIZ_FUNC_EXEC_TIME):
                logger.info(f"Discarding function as it did not run correctly or within "
                            f"{_MAX_VIZ_FUNC_EXEC_TIME} seconds.")
                continue

            #  Run some optimizations
            logger.info("Running Variable Name Optimization")
            var_optimized_code = optimize_vars(code, [], {arg: df.copy() for arg, df in df_args.items()})
            logger.info(f"Finished Running Variable Name Optimization:\n{var_optimized_code}")
            code = var_optimized_code

            #  Remove unnecessary statements that do not affect the visualization.
            #  Currently, the minimization being employed is a simplified version of the one used in VizSmith.
            logger.info("Running Minimization")
            minimized_code = minimize_code(code, [], df_args, timeout_per_run=_MAX_VIZ_FUNC_EXEC_TIME)
            logger.info(f"Finished Running Minimization:\n{minimized_code}")
            code = minimized_code

            #  We lift the hard-coded column references to column parameters.
            code_ast = astlib.parse(code)
            logger.info("Extracting Column Parameters")
            code, col_args = self._create_col_parameters(code_ast, df_args)
            logger.info(f"Finished Extracting Column Parameters:\nCode:\n{code}\nArguments:\n{col_args}")

            #  We are all set. Save the output to an appropriate place.
            logger.info(f"Final Processed Visualization Function:\n{code}")

            #  Figure out the paths to store the input dataframes at.
            self.update_df_mappings(df_args, df_obj_id_to_df, df_obj_id_to_pkl_paths)

            viz_code.append(self.get_viz_code_json(code, df_args, col_args, df_obj_id_to_pkl_paths))

        # Prettify saved data and write to files
        self.write_output(viz_code, df_obj_id_to_pkl_paths, df_obj_id_to_df, output_dir_path)

    def _check_execution(self, code: str, pos_args: List[Any], kw_args: Dict[str, Any],
                         timeout: int = _MAX_VIZ_FUNC_EXEC_TIME) -> bool:
        try:
            fig = mpl_exec.run_viz_code_matplotlib_mp(code, pos_args=pos_args, kw_args=kw_args,
                                                      timeout=timeout, func_name='viz')
            return fig is not None
        except:
            return False

    def ignore_figure(self, obj_id, fig, fig_to_slicing_criteria):
        if obj_id not in fig_to_slicing_criteria:
            logger.debug("Ignoring figure with zero writes")
            #  There's no write to this figure, ignore.
            return True

        if len(fig.axes) == 0:
            logger.info("Ignoring figure with zero axes")
            #  Need to have at least one axis to be non-empty.
            return True
        return False

    def get_fig_to_slicing_criteria(self, trace, body_stmts):
        #  Get all the matplotlib objects seen during the course of execution of the notebook.
        mpl_objs = self.fig_detector.get_matplotlib_figure_objects()
        #  The hierarchical trace uses object-IDs to identify objects instead of directly storing them.
        #  So we create a map from figure object IDs to the figures themselves for convenience.
        obj_id_to_fig: Dict[int, plt.Figure] = {id(o): o for o in mpl_objs}
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
        return fig_to_slicing_criteria