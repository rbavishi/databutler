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
