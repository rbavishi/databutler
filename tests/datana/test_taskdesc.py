import itertools
import unittest

from databutler.datana.generic.training import code2nl, few_shot, nl2code


class TaskDescTests(unittest.TestCase):
    def test_get_nl(self):
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
        #  We pass in a "header" task description to the nl generator as well
        generated_nl = nl_generator.get_nl(few_shot_examples, target_code,
                                        task_desc=["We attempt to generate descriptions of code as follows."], num_results=1)
        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))

    def test_get_code(self):
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
        #  We pass in a "header" task description to the code generator as well
        generated_nl = nl_generator.get_nl(few_shot_examples, target_code, num_results=1)
        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl,
                                task_desc=["We attempt to write blocks of code with similar to the given description."])

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))

    def test_get_nl_bullets(self):
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

        #  We pass in a "header" task descriptor to the nl bullet generator
        task_desc = ["We attempt to generate descriptions of code as follows."]

        #  The generated NL should be something like "A function to bitwise-AND two numbers"
        generated_nl = list(itertools.islice(nl_generator.get_nl_bullets(few_shot_examples, target_code, task_desc), 2))
        self.assertIsInstance(generated_nl, list)
        self.assertEqual(2, len(generated_nl))

        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))

    # Same as test_get_nl and test_get_code, except that it passes task descriptions into both functions.
    def test_combined(self):
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
        #  We pass in a "header" task description to the code generator as well
        generated_nl = nl_generator.get_nl(few_shot_examples, target_code,
                                task_desc=["We attempt to generate descriptions of blocks of code as follows:"], num_results=1)
        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl,
                                task_desc=["We attempt to write blocks of code with similar to the given description."])

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5 & 10, ctx['f'](10, 5))
        self.assertEqual(5 & 20, ctx['f'](5, 20))
