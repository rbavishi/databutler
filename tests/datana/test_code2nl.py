import itertools
import unittest

from databutler.datana.generic.training import code2nl, few_shot, nl2code


class Code2NLTests(unittest.TestCase):
    def test_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to add two numbers",
                code=(
                    "def f(a, b):\n"
                    "    return a + b"
                )
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to multiply two numbers",
                code=(
                    "def f(a, b):\n"
                    "    return a * b"
                )
            ),
        ]

        #  Since we can't check against the NL directly, we shall use bidirectional consistency.
        #  That is, the NL-to=Code component must work with the generated description.
        target_code = (
            "def f(x, y):"
            "    return x & y"
        )

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to bitwise-AND two numbers"
        generated_nl = nl_generator.get_nl(few_shot_examples, target_code, num_results=1)
        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))

    def test_2(self):
        #  Same as test_1, but we use 2 bullet points.
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "A function that takes two integers as input",
                    "Returns their sum"
                ],
                code=(
                    "def f(a, b):\n"
                    "    return a + b"
                )
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "A function that takes two integers as input",
                    "Returns their product"
                ],
                code=(
                    "def f(a, b):\n"
                    "    return a * b"
                )
            ),
        ]

        #  Since we can't check against the NL directly, we shall use bidirectional consistency.
        #  That is, the NL-to=Code component must work with the generated description.
        target_code = (
            "def f(x, y):"
            "    return x & y"
        )

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to bitwise-AND two numbers"
        generated_nl = list(itertools.islice(nl_generator.get_nl_bullets(few_shot_examples, target_code), 2))
        self.assertIsInstance(generated_nl, list)
        self.assertEqual(2, len(generated_nl))

        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))