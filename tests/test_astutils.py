import textwrap
import unittest

from databutler.pat import astlib


class TestASTUtils(unittest.TestCase):
    def test_1(self):
        source = """
        import a, b, c as d, e as f  # expect to keep: a, c as d
        from g import h, i, j as k, l as m  # expect to keep: h, j as k
        from n import o  # expect to be removed entirely

        a()

        def fun():
            d()
            baz, fooz = [i for i in [1,2,3,4]]
            print(c)

        class Cls:
            att = h.something

            def __new__(self) -> "Cls":
                var = k.method()
                func_undefined(var_undefined)
        """

        source = textwrap.dedent(source)
        ast = astlib.parse(source)

    def test_2(self):
        source = """
        a = 2
        a += 3
        """

        source = textwrap.dedent(source)
        ast = astlib.parse(source)
        defs, accesses = astlib.get_definitions_and_accesses(ast)
        self.assertEqual(2, len(defs))
        self.assertEqual(1, len(accesses))
