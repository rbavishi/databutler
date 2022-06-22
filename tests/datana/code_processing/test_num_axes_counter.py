import textwrap
import unittest

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.viz.corpus.code_processors import VizMplAxesCounter
from databutler.utils import multiprocess
from databutler.utils.libversioning import modified_lib_env


def _runner(func: DatanaFunction):
    #  Need to keep this outer-level to be able to run with pebble.concurrent.
    #  See https://github.com/noxdafox/pebble/issues/80
    with modified_lib_env("seaborn", "0.11.0"):
        normalizer = VizMplAxesCounter()
        return normalizer.run(func)


class NumAxesCounter(unittest.TestCase):
    def test_1(self):
        code = textwrap.dedent(
            """
        def visualization(df0_records, col0, col1, col2):
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            df0 = pd.DataFrame.from_records(df0_records)
            (fig, axes) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
            sns.distplot(a=df0[col0], color='b', ax=axes[0])
            sns.distplot(a=df0[col1], color='r', ax=axes[1])
            sns.distplot(a=df0[col2], color='y', ax=axes[2])
        """
        )

        #  Passing in records instead of dataframe as different Pandas versions cause havoc.
        df_records = [
            {"Age": 20, "Salary": 5000, "Rank": 1},
            {"Age": 22, "Salary": 1000, "Rank": 2},
            {"Age": 21, "Salary": 2000, "Rank": 3},
        ]

        datana_func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="visualization",
            pos_args=[df_records, "Age", "Salary", "Rank"],
            kw_args=None,
        )

        new_d_func = multiprocess.run_func_in_process(_runner, datana_func)
        self.assertEqual(
            3,
            new_d_func.metadata[VizMplAxesCounter.get_processor_metadata_key()][
                "num_axes"
            ],
        )

    def test_2(self):
        code = textwrap.dedent(
            """
        def visualization(df0_records, col0):
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            df0 = pd.DataFrame.from_records(df0_records)
            sns.distplot(a=df0[col0], color='b')
        """
        )

        #  Passing in records instead of dataframe as different Pandas versions cause havoc.
        df_records = [
            {"Age": 20, "Salary": 5000, "Rank": 1},
            {"Age": 22, "Salary": 1000, "Rank": 2},
            {"Age": 21, "Salary": 2000, "Rank": 3},
        ]

        datana_func = DatanaFunction(
            code_str=code,
            uid="test",
            func_name="visualization",
            pos_args=[df_records, "Age"],
            kw_args=None,
        )

        new_d_func = multiprocess.run_func_in_process(_runner, datana_func)
        self.assertEqual(
            1,
            new_d_func.metadata[VizMplAxesCounter.get_processor_metadata_key()][
                "num_axes"
            ],
        )
