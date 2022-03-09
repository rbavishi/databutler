from typing import Iterator, Tuple, Dict

import attrs
import plotly.graph_objs

from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import Instrumenter, ExprWrappersGenerator, ExprWrapper, Instrumentation
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType


@attrs.define(eq=False, repr=False)
class PlotlyFigureExprDetector(ExprWrappersGenerator):
    _found_exprs: Dict[astlib.Call, plotly.graph_objs.Figure] = attrs.field(init=False, factory=dict)

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
                self._found_exprs[call_expr] = value

            return value

        return wrapper

    def get_found_figures(self):
        return self._found_exprs.items()


@attrs.define(eq=False, repr=False)
class PlotlyMiner(BaseExecutor):
    @classmethod
    @register_runner(name="plotly_miner")
    def mining_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        plotly_fig_detector = PlotlyFigureExprDetector()

        instrumentation = Instrumentation.from_generators(plotly_fig_detector)
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

        for call_expr, obj in plotly_fig_detector.get_found_figures():
            print("Found Expr", astlib.to_code(call_expr))
