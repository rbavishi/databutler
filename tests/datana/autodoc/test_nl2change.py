import textwrap
import unittest

from databutler.datana.generic.autodoc import few_shot, nl2change


class NL2CodeChangeTests(unittest.TestCase):
    def test_full_new_code_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeChangeAndNL(
                nl="Change the operator to multiply instead of add",
                old_code=(
                    "def f(a, b):\n"
                    "    return a + b"
                ),
                new_code=(
                    "def f(a, b):\n"
                    "    return a * b"
                )
            ),
            few_shot.FewShotExampleCodeChangeAndNL(
                nl="Change the operator to add instead of multiply",
                old_code=(
                    "def f(a, b):\n"
                    "    return a * b"
                ),
                new_code=(
                    "def f(a, b):\n"
                    "    return a + b"
                )
            ),
        ]

        target_old_code = (
            "def f(a, b):\n"
            "    return a / b"
        )
        target_nl = "Change the operator to minus instead of divide"
        generator = nl2change.NatLangToNewCode()
        task = nl2change.NatLangToCodeChangeTask(few_shot_examples=few_shot_examples, target_old_code=target_old_code,
                                                 target_nl=target_nl)
        generated_code = generator.get_changed_code(task)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5, ctx['f'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['f'](5, 20))  # 5 - 20 = -15

        #  Try with output prefixes. Use specific variable names.
        output_prefix = "def func(x, y):\n"

        task = nl2change.NatLangToCodeChangeTask(few_shot_examples=few_shot_examples, target_old_code=target_old_code,
                                                 target_nl=target_nl, output_prefix=output_prefix)
        generated_code = generator.get_changed_code(task)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertIn('func', ctx.keys())
        self.assertEqual(5, ctx['func'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['func'](5, 20))  # 5 - 20 = -15

    def test_stmt_blanks_1(self):
        #  Check if the blanks are being added correctly
        gen = nl2change.NatLangToStmtBlanks()

        #  Example 1

        old_1 = textwrap.dedent("""
        def func(n: int):
            return n + 1
        """)
        new_1 = textwrap.dedent("""
        def func(n: int):
            return n + 2
        """)
        target_1_code = textwrap.dedent(f"""
        def func(n: int):
            {gen.blank_term}-1
        """)
        target_1_ans = ["return n + 2"]

        #  Example 2

        old_2 = textwrap.dedent("""
        def func(n: int):
            return n + 1
        """)
        new_2 = textwrap.dedent("""
        def func(n: int):
            ctr = n + 2
            return ctr
        """)
        target_2_code = textwrap.dedent(f"""
        def func(n: int):
            {gen.blank_term}-1
            {gen.blank_term}-2
        """)
        target_2_ans = ["ctr = n + 2", "return ctr"]

        for old, new, target_code, target_ans in [(old_1, new_1, target_1_code, target_1_ans),
                                                  (old_2, new_2, target_2_code, target_2_ans)]:
            blanked_code, ans = gen._create_blanks_and_answers(old, new)
            self.assertEqual(target_code, blanked_code)
            self.assertListEqual(target_ans, ans)
