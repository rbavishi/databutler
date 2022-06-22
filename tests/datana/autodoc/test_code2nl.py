import itertools
import unittest

from databutler.datana.generic.autodoc import code2nl, few_shot, nl2code


class Code2NLTests(unittest.TestCase):
    def test_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to add two numbers",
                code=("def f(a, b):\n" "    return a + b"),
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to multiply two numbers",
                code=("def f(a, b):\n" "    return a * b"),
            ),
        ]

        #  Since we can't check against the NL directly, we shall use bidirectional consistency.
        #  That is, the NL-to=Code component must work with the generated description.
        target_code = "def f(x, y):" "    return x & y"

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to bitwise-AND two numbers"
        code2nl_task = code2nl.CodeToNatLangTask(
            few_shot_examples=few_shot_examples, target_code=target_code
        )
        generated_nl = nl_generator.get_nl(code2nl_task, num_results=1)

        nl2code_task = nl2code.NatLangToCodeTask(
            few_shot_examples=few_shot_examples, target_nl=generated_nl
        )
        regenerated_code = code_generator.get_code(nl2code_task)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn("f", ctx.keys())
        self.assertEqual(5 & 10, ctx["f"](10, 5))
        self.assertEqual(5 & 20, ctx["f"](5, 20))

    def test_2(self):
        #  Same as test_1, but we use 2 bullet points.
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl=["A function that takes two integers as input", "Returns their sum"],
                code=("def f(a, b):\n" "    return a + b"),
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "A function that takes two integers as input",
                    "Returns their product",
                ],
                code=("def f(a, b):\n" "    return a * b"),
            ),
        ]

        #  Since we can't check against the NL directly, we shall use bidirectional consistency.
        #  That is, the NL-to=Code component must work with the generated description.
        target_code = "def f(x, y):" "    return x & y"

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function that takes two integers as input" and
        #  "Returns the bitwise-AND two numbers"
        code2nl_task = code2nl.CodeToNatLangTask(
            few_shot_examples=few_shot_examples, target_code=target_code
        )
        generated_nl = list(
            itertools.islice(nl_generator.get_nl_bullets(code2nl_task), 2)
        )
        self.assertIsInstance(generated_nl, list)
        self.assertEqual(2, len(generated_nl))

        nl2code_task = nl2code.NatLangToCodeTask(
            few_shot_examples=few_shot_examples, target_nl=generated_nl
        )
        regenerated_code = code_generator.get_code(nl2code_task)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn("f", ctx.keys())
        self.assertEqual(5 & 10, ctx["f"](10, 5))
        self.assertEqual(5 & 20, ctx["f"](5, 20))

    def test_task_descriptions_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to add two numbers",
                code=("def f(a, b):\n" "    return a + b"),
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to multiply two numbers",
                code=("def f(a, b):\n" "    return a * b"),
            ),
        ]

        #  Since we can't check against the NL directly, we shall use bidirectional consistency.
        #  That is, the NL-to=Code component must work with the generated description.
        target_code = "def f(x, y):" "    return x & y"

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to bitwise-AND two numbers"
        #  We pass in a "header" task description to the nl generator as well
        task_description_nl = "We attempt to generate descriptions of code as follows."
        code2nl_task = code2nl.CodeToNatLangTask(
            few_shot_examples=few_shot_examples,
            target_code=target_code,
            task_description=task_description_nl,
        )
        generated_nl = nl_generator.get_nl(code2nl_task, num_results=1)

        task_description_code = (
            "We attempt to write blocks of code with similar to the given description."
        )
        nl2code_task = nl2code.NatLangToCodeTask(
            few_shot_examples=few_shot_examples,
            target_nl=generated_nl,
            task_description=task_description_code,
        )
        regenerated_code = code_generator.get_code(nl2code_task)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn("f", ctx.keys())
        self.assertEqual(5 & 10, ctx["f"](10, 5))
        self.assertEqual(5 & 20, ctx["f"](5, 20))

        #  Check if the task descriptions are indeed included in the prompt
        prompt_nl = nl_generator._create_completion_prompt(code2nl_task)
        prompt_code = code_generator._create_completion_prompt(nl2code_task)
        self.assertIn(task_description_nl, prompt_nl)
        self.assertIn(task_description_code, prompt_code)

    def test_parallel_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to add two numbers",
                code=("def f(a, b):\n" "    return a + b"),
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl="A function to multiply two numbers",
                code=("def f(a, b):\n" "    return a * b"),
            ),
        ]

        target_codes = [
            ("def f(x, y):" "    return x & y"),
            ("def f(x, y):" "    return x | y"),
        ]

        nl_generator = code2nl.SimpleCodeToNatLang()
        code2nl_tasks = [
            code2nl.CodeToNatLangTask(
                few_shot_examples=few_shot_examples, target_code=target_code
            )
            for target_code in target_codes
        ]

        descriptions = nl_generator.parallel_get_nl(code2nl_tasks)
        self.assertEqual(2, len(descriptions))
        self.assertEqual(1, len(descriptions[0]))
        self.assertEqual(1, len(descriptions[1]))
        self.assertIn("bitwise and", descriptions[0][0].lower())
        self.assertIn("bitwise or", descriptions[1][0].lower())
