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
from scripts.mining.kaggle.execution.base import BaseExecutor

from scripts.mining.kaggle.execution.plotly_mining import utils

_MAX_VIZ_FUNC_EXEC_TIME = 5

class VizMiner(BaseExecutor):
    """
    Wrapper around miners for visualization code.
    """
    def __init__(self, collectorClasses: List[ExprWrappersGenerator] = []):
        self.collectorClasses = collectorClasses

    def mining_runner(self, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        #  A clock is critical in identifying dependencies.
        clock = LogicalClock()
        #  Trace instrumentation does the heavy-lifting of recording reads/writes, var. defs and their uses.
        trace_instrumentation = get_hierarchical_trace_instrumentation(clock=clock)
        #  Ready up the instrumentation for the matplotlib and df detectors.

        collectors = [CollectorClass() for CollectorClass in self.collectorClasses]
        instrumentations = [Instrumentation.from_generators(collector) for collector in self.collectorClasses]

        # plotly_instrumentation = Instrumentation.from_generators(plotly_fig_detector)
        # df_collector_instrumentation = Instrumentation.from_generators(df_collector)
        # col_collector_instrumentation = Instrumentation.from_generators(col_collector)
        # var_detection_instrumentation = Instrumentation.from_generators(var_detector)

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
                              fig_detector=plotly_fig_detector, var_detector=var_detector, output_dir_path=output_dir_path)

