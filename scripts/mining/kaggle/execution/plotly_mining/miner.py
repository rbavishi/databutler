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
class PlotlyMiner(BaseExecutor):
    @classmethod
    @register_runner(name="plotly_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        #  A clock is critical in identifying dependencies.
        clock = LogicalClock()
        #  Trace instrumentation does the heavy-lifting of recording reads/writes, var. defs and their uses.
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        #  Ready up the instrumentation for the matplotlib and df detectors.
        plotly_fig_detector = PlotlyFigureDetector()
        df_collector = ReadCsvDfCollector()
        col_collector = DfStrColumnsCollector()
        var_detector = PlotlyFigureVariableNameDetector()

        plotly_instrumentation = Instrumentation.from_generators(plotly_fig_detector)
        df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        col_collector_instrumentation = Instrumentation.from_generators(col_collector)
        var_detection_instrumentation = Instrumentation.from_generators(var_detector)

        instrumentation = (trace_instrumentation | plotly_instrumentation |
                            df_collector_instrumentation |col_collector_instrumentation |
                            var_detection_instrumentation
                            )

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

        trace = trace_instrumentation.get_hierarchical_trace()

        #  Use the trace, matplotlib figure and df detectors to extract visualization code.
        cls._extract_viz_code(code_ast, trace, df_collector=df_collector, col_collector=col_collector,
                              fig_detector=plotly_fig_detector, var_detector=var_detector ,output_dir_path=output_dir_path)

    @classmethod
    def _extract_viz_code(cls, code_ast: astlib.AstNode, trace: HierarchicalTrace,
                          df_collector: ReadCsvDfCollector,
                          col_collector: DfStrColumnsCollector,
                          fig_detector: PlotlyFigureDetector,
                          var_detector: PlotlyFigureVariableNameDetector,
                          output_dir_path: str):

        id_to_var_names = var_detector.get_found_vars()

        #  The hierarchical trace uses object-IDs to identify objects instead of directly storing them.
        #  So we create a map from figure object IDs to the figures themselves for convenience.
        obj_id_to_fig: Dict[int, plt.Figure] = fig_detector.get_found_figures()

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
        viz_code: List[Dict] = []
        df_obj_id_to_pkl_paths: Dict[int, str] = {}
        df_obj_id_to_df: Dict[int, pd.DataFrame] = {}
        logger.info("Extracting Visualization Code")
        for obj_id, fig in obj_id_to_fig.items():
            if obj_id not in fig_to_slicing_criteria:
                logger.debug("Ignoring figure with zero writes")
                #  There's no write to this figure, ignore.
                continue

            #  Extract the slice as a list of trace items, whose corresponding ast node are the ones we will
            #  use to build up the body of our visualization.
            criteria = fig_to_slicing_criteria[obj_id]
            viz_slice: List[TraceItem] = cls._get_slice(trace, criteria, body_stmts)
            viz_body = [item.ast_node for item in sorted(viz_slice, key=lambda x: x.start_time)]

            # We add a return at the end of the body, to return the figure object.
            if obj_id in id_to_var_names:
                var_name = id_to_var_names[obj_id]
                return_stmt = astlib.create_return(astlib.Name(var_name))
                viz_body.append(return_stmt)
            else:
                logger.info(f"No variable to return detected")
                continue

            new_body = astlib.prepare_body(viz_body)
            candidate = astlib.update_stmt_body(code_ast, new_body)

            logger.info(f"Extracted Raw Visualization Slice:\n{astlib.to_code(candidate)}")

            #  We want to tuck the slice under a function that could be used for other data-frames.
            #  First, we replace all the read_csv calls with dataframe arguments.
            #  We will consult the read_csv df collector for this.
            all_nodes = set(astlib.walk(candidate))
            to_replace: List[Tuple[astlib.AstNode, pd.DataFrame]] = []
            for node, df in df_collector.get_collected_dfs().items():
                if node in all_nodes:
                    to_replace.append((node, df))

            #  The mapping df_args will represent the arguments to be provided to the function
            #  to recreate the visualization.
            if len(to_replace) == 1:
                var_name = "_df"
                replacements = {to_replace[0][0]: astlib.create_name_expr(var_name)}
                df_args = {var_name: to_replace[0][1]}
            else:
                var_prefix = "_df_"
                replacements = {node: astlib.create_name_expr(f"{var_prefix}{idx}")
                                for idx, (node, df) in enumerate(to_replace, 1)}
                df_args = {f"{var_prefix}{idx}": df for idx, (node, df) in enumerate(to_replace, 1)}

            #  With the arguments figured out, we can construct the desired function by creating a new function
            #  with the required signature, and making the slice the body of the function.
            candidate = astlib.with_deep_replacements(candidate, replacements)
            func_def = astlib.parse_stmt(f"def viz({', '.join(i.value for i in replacements.values())}):\n    pass")
            func_def = astlib.update_stmt_body(func_def, candidate.body)
            code = astlib.to_code(func_def)

            logger.info(f"Extracted Visualization Function:\n{code}")

            #  Check if it executes correctly, and finishes under a fixed timeout
            if not cls._check_execution(code, pos_args=[], kw_args=df_args, timeout=_MAX_VIZ_FUNC_EXEC_TIME):
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
            code, col_args = cls._create_col_parameters(code_ast, df_args, col_collector=col_collector)
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

        mining_output_dir = os.path.join(output_dir_path, cls.__name__)
        os.makedirs(mining_output_dir, exist_ok=True)

        code_artifact = {
            "viz_functions": viz_code,
            "module_versions": cls._get_module_versions(),
        }

        #  Prettify multiline string output in YAML.
        #  See https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data

        def str_presenter(dumper, data):
            if len(data.splitlines()) > 1:  # check for multiline string
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_presenter)

        logger.info("Dumping viz functions")
        with open(os.path.join(mining_output_dir, "viz_functions.yaml"), "w") as f:
            yaml.dump(code_artifact, f)

        logger.info("Dumping dataframes")
        for obj_id, pkl_path in df_obj_id_to_pkl_paths.items():
            df = df_obj_id_to_df[obj_id]
            df.to_pickle(os.path.join(mining_output_dir, pkl_path))

        logger.info("Finished Extraction")

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


    @classmethod
    def _check_execution(cls, code: str, pos_args: List[Any], kw_args: Dict[str, Any],
                         timeout: int = _MAX_VIZ_FUNC_EXEC_TIME) -> bool:
        try:
            fig = utils.run_viz_code_plotly_mp(code, pos_args=pos_args, kw_args=kw_args,
                                                        timeout=timeout, func_name='viz')
            return fig is not None
        except:
            return False

    @classmethod
    def _collect_col_attribute_access(cls, code_ast: astlib.AstNode, pos_args: List[Any], kw_args: Dict[str, Any]
                                      ) -> Dict[astlib.Attribute, str]:
        attr_access_collector = DfColAttrAccessCollector()
        instrumenter = Instrumenter(Instrumentation.from_generators(attr_access_collector))

        #  Run the instrumenter, and execute.
        new_ast, globs = instrumenter.process(code_ast)
        new_code = astlib.to_code(new_ast)
        exec(new_code, globs, globs)
        globs['viz'](*pos_args, **kw_args)

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
                            new_param = f"_col_{num_inp_df_cols_found}"
                        else:
                            num_other_df_cols_found += 1
                            new_param = f"_new_col_{num_other_df_cols_found}"

                        cols_to_params[col_name] = new_param

                    node_replacement_map[node] = astlib.create_name_expr(cols_to_params[col_name])

            elif isinstance(node, astlib.Attribute) and node in attr_accesses:
                #  It's an attribute-based column access. We'll replace it with a subscript access i.e. (df["col_name"])
                col_name = attr_accesses[node]
                if col_name not in cols_to_params:
                    if col_name in inp_df_cols:
                        num_inp_df_cols_found += 1
                        new_param = f"_col_{num_inp_df_cols_found}"
                    else:
                        num_other_df_cols_found += 1
                        new_param = f"_new_col_{num_other_df_cols_found}"

                    cols_to_params[col_name] = new_param

                node_replacement_map[node] = astlib.create_subscript_expr(node.value, [cols_to_params[col_name]])

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
        updated_func_def = astlib.parse_stmt(f"def viz({new_sig}):\n    pass")
        updated_func_def = astlib.update_stmt_body(updated_func_def,
                                                   list(astlib.prepare_body(list(astlib.iter_body_stmts(func_def)))))

        new_code = astlib.to_code(updated_func_def)
        col_arguments = {v: k for k, v in cols_to_params.items()}

        return new_code, col_arguments

    @classmethod
    def _get_module_versions(cls):
        result = {}
        for k, v in sys.modules.items():
            if "." not in k and not k.startswith("_"):
                if hasattr(v, "__version__") and isinstance(getattr(v, "__version__"), str):
                    result[k] = getattr(v, "__version__")

        return result