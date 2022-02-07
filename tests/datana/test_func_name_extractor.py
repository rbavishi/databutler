import textwrap
import unittest

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.viz.corpus.code_processors import VizMplFuncNameExtractor
from databutler.utils import multiprocess
from databutler.utils.libversioning import modified_lib_env


def _seaborn_runner(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        normalizer = VizMplFuncNameExtractor()
        return normalizer.run(func)


class FuncNameExtractorTests(unittest.TestCase):
    def test_seaborn_1(self):
        code = textwrap.dedent(
            """
            def func(n: int):
                import seaborn as sns
                assert sns.__version__ == "0.11.0"
                sns.distplot(list(range(1, n)))
            """
        )

        datana_func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        new_d_func = multiprocess.run_func_in_process(_seaborn_runner, datana_func)
        metadata_key = f"metadata-{VizMplFuncNameExtractor.get_processor_name()}"
        func_name_mappings = new_d_func.metadata[metadata_key]['func_name_mappings']

        #  Just one function call to check
        self.assertEqual(1, len(func_name_mappings))
        key = "sns.distplot(list(range(1, n)))"
        self.assertIn(key, func_name_mappings)
        self.assertTrue(func_name_mappings[key].startswith("seaborn."))
        self.assertTrue(func_name_mappings[key].endswith(".distplot"))
