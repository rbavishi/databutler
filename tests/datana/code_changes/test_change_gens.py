import textwrap
import unittest

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.code_changes import change
from databutler.datana.viz.corpus import code_processors, change_gens
from databutler.utils import multiprocess
from databutler.utils.libversioning import modified_lib_env
from databutler.utils import code as codeutils


def _seaborn_runner_func_name_extractor(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        extractor = code_processors.VizMplFuncNameExtractor()
        return extractor.run(func)


def _seaborn_runner_kw_normalizer(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        normalizer = code_processors.VizMplKeywordArgNormalizer()
        return normalizer.run(func)


class CodeChangeGenTests(unittest.TestCase):
    def test_viz_mpl_1(self):
        code = textwrap.dedent(
            """
            def func(n: int):
                import seaborn as sns
                assert sns.__version__ == "0.11.0"
                sns.distplot(list(range(1, n)), kde=False)
            """
        )

        target = textwrap.dedent(
            """
            def func(n: int):
                import seaborn as sns
                assert sns.__version__ == "0.11.0"
                sns.distplot(list(range(1, n)))
            """
        )

        func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        func = multiprocess.run_func_in_process(_seaborn_runner_kw_normalizer, func)
        func = multiprocess.run_func_in_process(_seaborn_runner_func_name_extractor, func)

        eligible_funcs = {('seaborn', 'distplot')}
        eligible_kws = {('seaborn', 'distplot'): {'kde'}}

        c_gen = change_gens.VizMplConstKwArgRemover(eligible_funcs, eligible_kws)
        changes = c_gen.gen_changes(func)

        self.assertEqual(1, len(changes))
        gen_code = change.SimpleAstLibRemovalChange.apply_changes(code, changes)

        self.assertEqual(codeutils.normalize_code(target),
                         codeutils.normalize_code(gen_code))

        #  Also check that no changes are generated if the eligible dictionaries do not allow it..
        eligible_funcs = {('seaborn', 'heatmap')}
        eligible_kws = {('seaborn', 'heatmap'): {'annot'}}

        c_gen = change_gens.VizMplConstKwArgRemover(eligible_funcs, eligible_kws)
        changes = c_gen.gen_changes(func)

        self.assertEqual(0, len(changes))

    def test_viz_mpl_2(self):
        code = textwrap.dedent(
            """
            def func(n: int):
                import seaborn as sns
                from scipy.stats import norm
                assert sns.__version__ == "0.11.0"
                sns.distplot(list(range(1, n)), kde=False, rug=True, fit=norm)
            """
        )

        #  fit=norm should not be removed by a const-kw remover
        target = textwrap.dedent(
            """
            def func(n: int):
                import seaborn as sns
                from scipy.stats import norm
                assert sns.__version__ == "0.11.0"
                sns.distplot(list(range(1, n)), fit=norm)
            """
        )

        func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="func",
            pos_args=[10],
            kw_args=None,
        )

        func = multiprocess.run_func_in_process(_seaborn_runner_kw_normalizer, func)
        func = multiprocess.run_func_in_process(_seaborn_runner_func_name_extractor, func)

        eligible_funcs = {('seaborn', 'distplot')}
        #  Putting fit in should still not make a difference as it's not a constant.
        eligible_kws = {('seaborn', 'distplot'): {'kde', 'fit', 'rug'}}

        c_gen = change_gens.VizMplConstKwArgRemover(eligible_funcs, eligible_kws)
        changes = c_gen.gen_changes(func)

        self.assertEqual(2, len(changes))
        gen_code = change.SimpleAstLibRemovalChange.apply_changes(code, changes)
