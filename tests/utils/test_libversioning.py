import unittest

from databutler.utils import multiprocess
from databutler.utils.libversioning import modified_lib_env


def _run_code(code: str, lib_name: str, version: str):
    with modified_lib_env(lib_name, version):
        exec(code)


class LibVersioningTests(unittest.TestCase):
    def test_pandas_1(self):
        #  df.ftypes was removed in Pandas 1.* but was only deprecated in 0.25.0
        code = (
            "import sys; print(sys.path)\n"
            "import pandas as pd\n"
            "assert pd.__version__ == '0.25.1'"
        )

        multiprocess.run_func_in_process(_run_code, code, "pandas", "0.25.1")

    def test_pandas_2(self):
        #  df.ftypes was removed in Pandas 1.* but was only deprecated in 0.25.0
        code = (
            "import sys; print(sys.path)\n"
            "import pandas as pd\n"
            "df = pd.DataFrame([[1, 2], [3, 4]])\n"
            "df.ftypes"
        )

        multiprocess.run_func_in_process(_run_code, code, "pandas", "0.25.1")


