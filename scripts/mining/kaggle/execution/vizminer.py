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
