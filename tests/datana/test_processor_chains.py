import textwrap
import unittest

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessorChain
from databutler.datana.viz.corpus.code_processors import VizMplKeywordArgNormalizer, VizMplVarNameOptimizer
from databutler.utils import multiprocess, code as codeutils
from databutler.utils.libversioning import modified_lib_env


def _seaborn_runner(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        processor = DatanaFunctionProcessorChain(
            processors=[
                VizMplKeywordArgNormalizer(),
                VizMplVarNameOptimizer(),
            ]
        )

        return processor.run(func)


class CodeProcessorChainTests(unittest.TestCase):
    def test_kw_normalizer_plus_var_name_optimizer_1(self):
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
                df0 = pd.DataFrame.from_records(data=df0_records)
                df0 = df0.dropna()
                sns.distplot(a=df0[col1])
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
        self.assertEqual(codeutils.normalize_code(target_code),
                         codeutils.normalize_code(new_d_func.code_str))
