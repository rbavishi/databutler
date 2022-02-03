from typing import List, Any, Dict

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessor
from databutler.datana.generic.corpus.processing.func_name_extractor import FuncNameExtractor
from databutler.datana.generic.corpus.processing.keyword_normalization import KeywordArgNormalizer
from databutler.datana.generic.corpus.processing.var_optimization import VarNameOptimizer
from databutler.datana.viz.utils import mpl_exec


@attrs.define(eq=False, repr=False)
class VizMplKeywordArgNormalizer(KeywordArgNormalizer):
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        mpl_exec.run_viz_code_matplotlib(
            code=func_code,
            pos_args=pos_args,
            kw_args=kw_args,
            func_name=func_name,
            other_globals=global_ctx
        )

    @classmethod
    def get_processor_name(cls) -> str:
        return f"viz-mpl-{super().get_processor_name()}"


@attrs.define(eq=False, repr=False)
class VizMplFuncNameExtractor(FuncNameExtractor):
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        mpl_exec.run_viz_code_matplotlib(
            code=func_code,
            pos_args=pos_args,
            kw_args=kw_args,
            func_name=func_name,
            other_globals=global_ctx
        )

    @classmethod
    def get_processor_name(cls) -> str:
        return f"viz-mpl-{super().get_processor_name()}"


@attrs.define(eq=False, repr=False)
class VizMplVarNameOptimizer(VarNameOptimizer):
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        mpl_exec.run_viz_code_matplotlib(
            code=func_code,
            pos_args=pos_args,
            kw_args=kw_args,
            func_name=func_name,
            other_globals=global_ctx
        )

    @classmethod
    def get_processor_name(cls) -> str:
        return f"viz-mpl-{super().get_processor_name()}"
