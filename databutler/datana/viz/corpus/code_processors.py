from typing import List, Any, Dict

import attrs

from databutler.datana.generic.corpus.processing.keyword_normalization import KeywordArgNormalizer
from databutler.datana.viz.utils import mpl_exec


@attrs.define(eq=False, repr=False)
class VizKeywordArgNormalizer(KeywordArgNormalizer):
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        mpl_exec.run_viz_code_matplotlib(
            code=func_code,
            pos_args=pos_args,
            kw_args=kw_args,
            func_name=func_name,
            other_globals=global_ctx
        )
