import ast
import textwrap
import unittest

from databutler.pat import astlib
from databutler.pat.analysis.type_analysis.inference import run_mypy


class TypeInferenceTests(unittest.TestCase):
    def test_simple_1(self):
        code = textwrap.dedent("""
        import pandas as pd
        col1 = "Survived"
        df = pd.read_csv("titanic.csv")
        df.groupby([col1], as_index=False)
        print(df)
        """)
        src_ast, inferred_types = run_mypy(code)
        for node in astlib.walk(src_ast):
            if astlib.expr_is_evaluated(node, src_ast):
                if isinstance(node, astlib.Name) and node.value == 'df':
                    self.assertEquals("pandas.core.frame.DataFrame", inferred_types[node].type_json)

    def test_simple_2(self):
        code = textwrap.dedent("""
        a = sum(i for i in range(10))
        print(f'abcd{a}abcd')
        sum(*[[1, 2, 3]])
        """)

        src_ast, inferred_types = run_mypy(code)
        for node, typ in inferred_types.items():
            print(astlib.to_code(node), typ)
