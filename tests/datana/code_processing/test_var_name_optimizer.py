import textwrap
import unittest
from typing import List, Any, Dict

import pandas as pd

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.var_optimization import (
    VarNameOptimizer,
)
from databutler.datana.viz.corpus.code_processors import VizMplVarNameOptimizer
from databutler.utils import code as codeutils, multiprocess
from databutler.utils.libversioning import modified_lib_env


def _seaborn_runner(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        normalizer = VizMplVarNameOptimizer()
        return normalizer.run(func)


class VarNameOptimizerTests(unittest.TestCase):
    def test_builtin_1(self):
        orig_code = textwrap.dedent(
            """
            def func(n: int):
                t = n + 1
                t_str = str(n)
                t_str_1 = t_str + "0"
                s = t + 1
                v = s + 1
                return v + int(t_str_1)
            """
        )

        target_code = textwrap.dedent(
            """
            def func(n: int):
                t = n + 1
                t_str = str(n)
                t_str = t_str + "0"
                n = t + 1
                n = n + 1
                return n + int(t_str)
            """
        )

        datana_func = DatanaFunction(
            code_str=orig_code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        class TestOptimizer(VarNameOptimizer):
            def _run_function_code(
                self,
                func_code: str,
                func_name: str,
                pos_args: List[Any],
                kw_args: Dict[str, Any],
                global_ctx: Dict[str, Any],
            ) -> Any:
                ctx = global_ctx.copy()
                exec(func_code, ctx)
                ctx[func_name](*pos_args, **kw_args)

        normalizer = TestOptimizer()
        new_d_func = normalizer.run(datana_func)
        self.assertEqual(
            codeutils.normalize_code(target_code),
            codeutils.normalize_code(new_d_func.code_str),
        )

    def test_builtin_2(self):
        orig_code = textwrap.dedent(
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

        #  An assignment (t = n) should be gotten rid of.
        target_code = textwrap.dedent(
            """
            def func(n: int):
                t_str = str(n)
                t_str = t_str + "0"
                n = n + 1
                n = n + 1
                return n + int(t_str)
            """
        )

        datana_func = DatanaFunction(
            code_str=orig_code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        class TestOptimizer(VarNameOptimizer):
            def _run_function_code(
                self,
                func_code: str,
                func_name: str,
                pos_args: List[Any],
                kw_args: Dict[str, Any],
                global_ctx: Dict[str, Any],
            ) -> Any:
                ctx = global_ctx.copy()
                exec(func_code, ctx)
                ctx[func_name](*pos_args, **kw_args)

        normalizer = TestOptimizer()
        new_d_func = normalizer.run(datana_func)
        self.assertEqual(
            codeutils.normalize_code(target_code),
            codeutils.normalize_code(new_d_func.code_str),
        )

    def test_builtin_3(self):
        orig_code = textwrap.dedent(
            """
            def func(n: int):
                for _ in range(5):
                    s = n + 1
                    v = s + 1
                    g = v
                return g
            """
        )

        #  For loops should be handled correctly. The only renaming that should happen is g -> v
        target_code = textwrap.dedent(
            """
            def func(n: int):
                for _ in range(5):
                    s = n + 1
                    v = s + 1
                return v
            """
        )

        datana_func = DatanaFunction(
            code_str=orig_code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        class TestOptimizer(VarNameOptimizer):
            def _run_function_code(
                self,
                func_code: str,
                func_name: str,
                pos_args: List[Any],
                kw_args: Dict[str, Any],
                global_ctx: Dict[str, Any],
            ) -> Any:
                ctx = global_ctx.copy()
                exec(func_code, ctx)
                ctx[func_name](*pos_args, **kw_args)

        normalizer = TestOptimizer()
        new_d_func = normalizer.run(datana_func)
        self.assertEqual(
            codeutils.normalize_code(target_code),
            codeutils.normalize_code(new_d_func.code_str),
        )

    def test_builtin_4(self):
        orig_code = textwrap.dedent(
            """
            def func(n: int):
                a = 10
                b = "a"
                c = [1, 2]
                d = {"a": "b"}
                e = {3, 4}
                f = (9, 10)
                
                return [n, a, b, c, d, e, f]
            """
        )

        #  Constant propagation should happen for all variables a to f.
        target_code = textwrap.dedent(
            """
            def func(n: int):
                return [n, 10, "a", [1, 2], {"a": "b"}, {3, 4}, (9, 10)]
            """
        )

        datana_func = DatanaFunction(
            code_str=orig_code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        class TestOptimizer(VarNameOptimizer):
            def _run_function_code(
                self,
                func_code: str,
                func_name: str,
                pos_args: List[Any],
                kw_args: Dict[str, Any],
                global_ctx: Dict[str, Any],
            ) -> Any:
                ctx = global_ctx.copy()
                exec(func_code, ctx)
                ctx[func_name](*pos_args, **kw_args)

        normalizer = TestOptimizer()
        new_d_func = normalizer.run(datana_func)
        self.assertEqual(
            codeutils.normalize_code(target_code),
            codeutils.normalize_code(new_d_func.code_str),
        )

    def test_seaborn_1(self):
        orig_code = textwrap.dedent(
            """
            def func(df0_records, col1):
                import seaborn as sns
                assert sns.__version__ == "0.11.0"
                import pandas as pd
                df0 = pd.DataFrame.from_records(df0_records)
                df1 = df0.dropna()
                sns.distplot(df1[col1])
            """
        )

        target_code = textwrap.dedent(
            """
            def func(df0_records, col1):
                import seaborn as sns
                assert sns.__version__ == "0.11.0"
                import pandas as pd
                df0 = pd.DataFrame.from_records(df0_records)
                df0 = df0.dropna()
                sns.distplot(df0[col1])
            """
        )

        #  Passing in records instead of dataframe as different Pandas versions cause havoc.
        df_records = [
            {"Age": 10, "Rating": "A"},
            {"Age": 12, "Rating": "B"},
            {"Age": 11, "Rating": "C"},
        ]

        datana_func = DatanaFunction(
            code_str=orig_code,
            uid="test",
            func_name="func",
            pos_args=[df_records, "Age"],
            kw_args=None,
        )

        new_d_func = multiprocess.run_func_in_process(_seaborn_runner, datana_func)
        self.assertEqual(
            codeutils.normalize_code(target_code),
            codeutils.normalize_code(new_d_func.code_str),
        )
