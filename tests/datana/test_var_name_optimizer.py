import textwrap
import unittest
from typing import List, Any, Dict

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.var_optimization import VarNameOptimizer
from databutler.utils import code as codeutils, multiprocess
from databutler.utils.libversioning import modified_lib_env
#
#
# def _seaborn_runner(func: DatanaFunction):
#     #  Need to keep this outer-level to be able to run with pebble.concurrent.
#     #  See https://github.com/noxdafox/pebble/issues/80
#     with modified_lib_env("seaborn", "0.11.0"):
#         normalizer = VizKeywordArgNormalizer()
#         return normalizer.run(func)


class VarNameOptimizerTests(unittest.TestCase):
    def test_builtin_1(self):
        code = textwrap.dedent(
            """
            def func(n: int):
                t = n
                t_str = str(n)
                t_str_1 = t_str + "0"
                s = t + 1
                v = s + 1
                return v + int(t_str_1)
            """
        )

        datana_func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        class TestOptimizer(VarNameOptimizer):
            def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                                   global_ctx: Dict[str, Any]) -> Any:
                ctx = global_ctx.copy()
                exec(func_code, ctx)
                ctx[func_name](*pos_args, **kw_args)

        normalizer = TestOptimizer()
        new_d_func = normalizer.run(datana_func)

    # def test_seaborn_1(self):
    #     orig_code = textwrap.dedent(
    #         """
    #         def func(n: int):
    #             import seaborn as sns
    #             assert sns.__version__ == "0.11.0"
    #             sns.distplot(list(range(1, n)))
    #         """
    #     )
    #
    #     target_code = textwrap.dedent(
    #         """
    #         def func(n: int):
    #             import seaborn as sns
    #             assert sns.__version__ == "0.11.0"
    #             sns.distplot(a=list(range(1, n)))
    #         """
    #     )
    #
    #     datana_func = DatanaFunction(
    #         code_str=orig_code,
    #         uid="test",
    #         func_name="func",
    #         pos_args=[10],
    #         kw_args=None,
    #     )
    #
    #     new_d_func = multiprocess.run_func_in_process(_seaborn_runner, datana_func)
    #     self.assertEqual(codeutils.normalize_code(target_code),
    #                      codeutils.normalize_code(new_d_func.code_str))
