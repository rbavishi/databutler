import ast
import collections
import os
import sys
from typing import Sequence, Set, Dict, List, Tuple, Iterator, Any

import attrs
from libcst import AssignTarget
import pandas as pd
import yaml
import plotly
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
from scripts.mining.kaggle.execution.plotly_mining.minimizer import minimize_code
from scripts.mining.kaggle.execution.mpl_seaborn_mining.var_optimization import optimize_vars
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType

from scripts.mining.kaggle.execution.plotly_mining import utils

from scripts.mining.kaggle.execution.vizminer import VizMiner
from functools import reduce

_MAX_VIZ_FUNC_EXEC_TIME = 5

@attrs.define(eq=False, repr=False)
class PlotlyFigureDetector(ExprWrappersGenerator):
    """
    Captures all the plotly objects observed during the execution of the program.
    When using plotly, whenever a statement returns a pltly Figure graph object, a
    visualization has been generated. We save these, by using expression callbacks that are
    called after every call expression in the AST is evaluated.
    """
    _found_figures: Dict[int, plotly.graph_objs.Figure] = attrs.field(init=False, factory=dict)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for expr in self.iter_valid_exprs(ast_root):
            if isinstance(expr, astlib.Call):
                yield expr, ExprWrapper(
                    callable=self.gen_plotly_wrapper_expr(expr),
                    name=self.gen_wrapper_id(),
                )

    def gen_plotly_wrapper_expr(self, call_expr: astlib.Call):
        def wrapper(value):
            if isinstance(value, plotly.graph_objs.Figure):
                print(f'Figure detected: {id(value)}')
                self._found_figures[id(value)] = value
            return value
        return wrapper

    def get_found_figures(self):
        return self._found_figures

    def get_found_objects(self):
        return self._found_figures.values()

@attrs.define(eq=False, repr=False)
class PlotlyFigureVariableNameDetector(ExprWrappersGenerator):
    """
    Captures all the variables mapping to plotly objects seen during the execution of
    the program.
    """
    # mapping from object ids to variable names
    _found_vars : Dict[int, Sequence[AssignTarget]] = attrs.field(init=False, factory=dict)

    def gen_expr_wrappers_simple(self, ast_root: astlib.AstNode) -> Iterator[Tuple[astlib.BaseExpression, ExprWrapper]]:
        for node in astlib.walk(ast_root):
            if isinstance(node, astlib.Assign):
                right_hand_expr = node.value
                if isinstance(right_hand_expr, astlib.Call):
                    yield right_hand_expr, ExprWrapper(
                        callable=self.gen_plotly_wrapper_expr(node),
                        name=self.gen_wrapper_id()
                    )

    def gen_plotly_wrapper_expr(self, stmt: astlib.Assign):
        def wrapper(value):
            if isinstance(value, plotly.graph_objs.Figure):
                graph_obj_id = id(value)
                self._found_vars[graph_obj_id] = stmt.targets
            return value
        return wrapper

    def get_found_vars(self):
        id_to_var_name: Dict[int, str] = {}
        for graph_id, targets in self._found_vars.items():
            if len(targets) == 1:
                var_name = targets[0].target
                if isinstance(var_name, astlib.Name):
                    id_to_var_name[graph_id] = var_name.value
        return id_to_var_name


@attrs.define(eq=False, repr=False)
class PlotlyMiner(VizMiner):

    def __init__(self):
        super().__init__()
        self.fig_detector = PlotlyFigureDetector()
        self.var_detector = PlotlyFigureVariableNameDetector()
        self.plotly_collectors = [self.fig_detector, self.var_detector]
        self.collectors.extend(self.plotly_collectors)

    @classmethod
    @register_runner(name="plotly_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):

        miner = PlotlyMiner()
        instrumenter = miner.get_instrumenter()
        code_ast = miner.get_code_ast(source, source_type)

        #  Run the instrumenter, and execute.
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)

        trace = miner.trace_instrumentation.get_hierarchical_trace()
        miner.trace = trace

        #  Use the trace, matplotlib figure and df detectors to extract visualization code.
        miner._extract_viz_code(code_ast, trace, output_dir_path=output_dir_path)

    def _extract_viz_code(self, code_ast: astlib.AstNode, trace: HierarchicalTrace,
                          output_dir_path: str):

        body_stmts = self.get_body_stmts(code_ast)
        fig_to_slicing_criteria = self.get_fig_to_slicing_criteria(trace, body_stmts)

        #  For each non-empty figure, we'll perform slicing using the collected criteria,
        viz_code: List[Dict] = []
        df_obj_id_to_pkl_paths: Dict[int, str] = {}
        df_obj_id_to_df: Dict[int, pd.DataFrame] = {}

        logger.info("Extracting Visualization Code")
        id_to_var_names = self.var_detector.get_found_vars()
        obj_id_to_fig: Dict[int, plt.Figure] = self.fig_detector.get_found_figures()

        for obj_id, fig in obj_id_to_fig.items():
            if obj_id not in fig_to_slicing_criteria:
                logger.debug("Ignoring figure with zero writes")
                #  There's no write to this figure, ignore.
                continue

            #  Extract the slice as a list of trace items, whose corresponding ast node are the ones we will
            #  use to build up the body of our visualization.
            criteria = fig_to_slicing_criteria[obj_id]
            viz_body = self.get_viz_body(criteria, trace, body_stmts)

            # We add a return at the end of the body, to return the figure object.
            viz_body = self.add_return_fig_stmt(obj_id, viz_body, id_to_var_names)
            if viz_body is None: continue
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
            for df in df_args.values():
                if id(df) not in df_obj_id_to_pkl_paths:
                    df_obj_id_to_pkl_paths[id(df)] = f"df_{len(df_obj_id_to_pkl_paths) + 1}.pkl"
                    df_obj_id_to_df[id(df)] = df

            viz_code.append({
                "code": code,
                "df_args": {arg: df_obj_id_to_pkl_paths[id(df)] for arg, df in df_args.items()},
                "col_args": col_args,
            })

        # Prettify saved data and write to files
        self.write_output(viz_code, df_obj_id_to_pkl_paths, df_obj_id_to_df, output_dir_path)


    def _check_execution(self, code: str, pos_args: List[Any], kw_args: Dict[str, Any],
                         timeout: int = _MAX_VIZ_FUNC_EXEC_TIME) -> bool:
        try:
            fig = utils.run_viz_code_plotly_mp(code, pos_args=pos_args, kw_args=kw_args,
                                                        timeout=timeout, func_name='viz')
            return fig is not None
        except:
            return False

    def get_fig_to_slicing_criteria(self, trace, body_stmts):
        #  The hierarchical trace uses object-IDs to identify objects instead of directly storing them.
        #  So we create a map from figure object IDs to the figures themselves for convenience.
        obj_id_to_fig: Dict[int, plt.Figure] = self.fig_detector.get_found_figures()
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

    def add_return_fig_stmt(self, obj_id, viz_body, id_to_var_names):
        if obj_id in id_to_var_names:
            var_name = id_to_var_names[obj_id]
            return_stmt = astlib.create_return(astlib.Name(var_name))
            viz_body.append(return_stmt)
            return viz_body
        else:
            logger.info(f"No variable to return detected")
            return None