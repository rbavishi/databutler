import textwrap
import unittest

from databutler.datana.generic.corpus.code_changes import change
from databutler.utils import code as codeutils


class CodeChangeTests(unittest.TestCase):
    def test_native_ast_1(self):
        orig_code = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n, b=n-2) 
                return yet_another_func(s, b=n+2)
            """
        )

        ref1 = change.SimpleAstNodeRef(node_type="keyword", index=0)  # Should refer to b=n-2
        ref2 = change.SimpleAstNodeRef(node_type="keyword", index=1)  # Should refer to b=n+2
        ref3 = change.SimpleAstNodeRef(node_type="Assign", index=0)  # Should refer to s = another_func(s, b=n+2)

        change1 = change.SimpleAstRemovalChange(node_refs=[ref1], children=[])
        change2 = change.SimpleAstRemovalChange(node_refs=[ref2], children=[])
        change3 = change.SimpleAstRemovalChange(node_refs=[ref3], children=[change1, change2])

        res1 = change.SimpleAstRemovalChange.apply_changes(orig_code, [change1])
        res2 = change.SimpleAstRemovalChange.apply_changes(orig_code, [change2])
        res3 = change.SimpleAstRemovalChange.apply_changes(orig_code, [change3])

        target1 = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n) 
                return yet_another_func(s, b=n+2)
            """
        )

        target2 = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n, b=n-2) 
                return yet_another_func(s)
            """
        )

        #  Since ref2 is a child of ref3, it should also get removed.
        target3 = textwrap.dedent(
            """
            def func(n: int):
                return yet_another_func(s)
            """
        )

        self.assertEqual(codeutils.normalize_code(target1),
                         codeutils.normalize_code(res1))
        self.assertEqual(codeutils.normalize_code(target2),
                         codeutils.normalize_code(res2))
        self.assertEqual(codeutils.normalize_code(target3),
                         codeutils.normalize_code(res3))

    def test_astlib_ast_1(self):
        orig_code = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n, b=n-2) 
                return yet_another_func(s, b=n+2)
            """
        )

        ref1 = change.SimpleAstLibNodeRef(node_type="Arg", index=1)  # Should refer to b=n-2
        ref2 = change.SimpleAstLibNodeRef(node_type="Arg", index=3)  # Should refer to b=n+2
        ref3 = change.SimpleAstLibNodeRef(node_type="Assign", index=0)  # Should refer to s = another_func(s, b=n+2)

        change1 = change.SimpleAstLibRemovalChange(node_refs=[ref1], children=[])
        change2 = change.SimpleAstLibRemovalChange(node_refs=[ref2], children=[])
        change3 = change.SimpleAstLibRemovalChange(node_refs=[ref3], children=[change1, change2])

        res1 = change.SimpleAstLibRemovalChange.apply_changes(orig_code, [change1])
        res2 = change.SimpleAstLibRemovalChange.apply_changes(orig_code, [change2])
        res3 = change.SimpleAstLibRemovalChange.apply_changes(orig_code, [change3])

        target1 = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n) 
                return yet_another_func(s, b=n+2)
            """
        )

        target2 = textwrap.dedent(
            """
            def func(n: int):
                s = another_func(n, b=n-2) 
                return yet_another_func(s)
            """
        )

        #  Since ref2 is a child of ref3, it should also get removed.
        target3 = textwrap.dedent(
            """
            def func(n: int):
                return yet_another_func(s)
            """
        )

        self.assertEqual(codeutils.normalize_code(target1),
                         codeutils.normalize_code(res1))
        self.assertEqual(codeutils.normalize_code(target2),
                         codeutils.normalize_code(res2))
        self.assertEqual(codeutils.normalize_code(target3),
                         codeutils.normalize_code(res3))
        pass
