import unittest

from databutler.datana.training import nl2code


class NL2CodeTests(unittest.TestCase):
    def test_1(self):
        few_shot_examples = [
            nl2code.FewShotExample(
                nl="A function to add two numbers",
                code=(
                    "def f(a, b):\n"
                    "    return a + b"
                )
            ),
            nl2code.FewShotExample(
                nl="A function to multiply two numbers",
                code=(
                    "def f(a, b):\n"
                    "    return a * b"
                )
            ),
        ]

        target_nl = "A function to subtract two numbers"
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertEqual(5, ctx['f'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['f'](5, 20))  # 5 - 20 = -15

        #  Try with output prefixes. Use specific variable names.
        output_prefix = "def func(x, y):\n"
        generated_code = generator.get_code(few_shot_examples, target_nl, output_prefix)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertEqual(5, ctx['func'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['func'](5, 20))  # 5 - 20 = -15
