import itertools
import unittest

from databutler.datana.generic.autodoc import change2nl, few_shot, nl2change


class CodeChange2NLTests(unittest.TestCase):
    def test_full_code_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeChangeAndNL(
                nl="Change the operator to multiply instead of add",
                old_code=("def f(a, b):\n" "    return a + b"),
                new_code=("def f(a, b):\n" "    return a * b"),
            ),
            few_shot.FewShotExampleCodeChangeAndNL(
                nl="Change the operator to add instead of multiply",
                old_code=("def f(a, b):\n" "    return a * b"),
                new_code=("def f(a, b):\n" "    return a + b"),
            ),
        ]

        target_old_code = "def f(a, b):\n" "    return a / b"

        target_new_code = "def f(a, b):\n" "    return a - b"

        for nl_generator in [
            change2nl.FullCodeChangeToNatLang(),
            change2nl.DiffToNatLang(),
        ]:
            code_generator = nl2change.NatLangToNewCode()

            nl_task = change2nl.CodeChangeToNatLangTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_new_code=target_new_code,
            )
            generated_nl = nl_generator.get_nl(nl_task, num_results=1)[0]

            #  No notion of a ground-truth, but the model should be able to generate the code from it.
            code_task = nl2change.NatLangToCodeChangeTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_nl=generated_nl,
            )
            regenerated_code = code_generator.get_changed_code(code_task)

            #  Run the generated code to see if it does the right thing
            ctx = {}
            exec(regenerated_code, ctx)
            self.assertIn("f", ctx.keys())
            self.assertEqual(5, ctx["f"](10, 5))  # 10 - 5 = 5
            self.assertEqual(-15, ctx["f"](5, 20))  # 5 - 20 = -15

            #  Also test with other change-generation strategies
            code_generator = nl2change.NatLangToStmtBlanks()

            #  No notion of a ground-truth, but the model should be able to generate the code from it.
            target_blanked, _ = code_generator.create_blanks_and_answers(
                target_old_code,
                target_new_code,
                blank_word=code_generator.default_blank_word,
            )
            code_task = nl2change.NatLangToCodeChangeTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_blanked=target_blanked,
                target_nl=generated_nl,
            )
            regenerated_code = code_generator.get_changed_code(code_task)

            #  Run the generated code to see if it does the right thing
            ctx = {}
            exec(regenerated_code, ctx)
            self.assertIn("f", ctx.keys())
            self.assertEqual(5, ctx["f"](10, 5))  # 10 - 5 = 5
            self.assertEqual(-15, ctx["f"](5, 20))  # 5 - 20 = -15

    def test_full_code_2(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeChangeAndNL(
                nl=[
                    "Change the operator to multiply instead of add",
                    "Also print 'Hello'",
                ],
                old_code=("def f(a, b):\n" "    return a + b"),
                new_code=("def f(a, b):\n" "    print('Hello')\n" "    return a * b"),
            ),
            few_shot.FewShotExampleCodeChangeAndNL(
                nl=[
                    "Change the operator to add instead of multiply",
                    "Also print 'World'",
                ],
                old_code=("def f(a, b):\n" "    return a * b"),
                new_code=("def f(a, b):\n" "    print('World')\n" "    return a + b"),
            ),
        ]

        target_old_code = "def f(a, b):\n" "    return a / b"

        target_new_code = (
            "def f(a, b):\n" "    print('Hello World')\n" "    return a - b"
        )

        for nl_generator in [
            change2nl.FullCodeChangeToNatLang(),
            change2nl.DiffToNatLang(),
        ]:
            code_generator = nl2change.NatLangToNewCode()

            nl_task = change2nl.CodeChangeToNatLangTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_new_code=target_new_code,
            )
            generated_nl = list(
                itertools.islice(nl_generator.get_nl_bullets(nl_task), 2)
            )
            self.assertEqual(2, len(generated_nl))

            #  No notion of a ground-truth, but the model should be able to generate the code from it.
            code_task = nl2change.NatLangToCodeChangeTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_nl=generated_nl,
            )
            regenerated_code = code_generator.get_changed_code(code_task)

            #  Run the generated code to see if it does the right thing
            ctx = {}
            exec(regenerated_code, ctx)
            self.assertIn("f", ctx.keys())
            self.assertEqual(5, ctx["f"](10, 5))  # 10 - 5 = 5
            self.assertEqual(-15, ctx["f"](5, 20))  # 5 - 20 = -15

            #  Also test with other change-generation strategies
            code_generator = nl2change.NatLangToStmtBlanks()

            #  No notion of a ground-truth, but the model should be able to generate the code from it.
            target_blanked, _ = code_generator.create_blanks_and_answers(
                target_old_code,
                target_new_code,
                blank_word=code_generator.default_blank_word,
            )
            code_task = nl2change.NatLangToCodeChangeTask(
                few_shot_examples=few_shot_examples,
                target_old_code=target_old_code,
                target_blanked=target_blanked,
                target_nl=generated_nl,
            )
            regenerated_code = code_generator.get_changed_code(code_task)

            #  Run the generated code to see if it does the right thing
            ctx = {}
            exec(regenerated_code, ctx)
            self.assertIn("f", ctx.keys())
            self.assertEqual(5, ctx["f"](10, 5))  # 10 - 5 = 5
            self.assertEqual(-15, ctx["f"](5, 20))  # 5 - 20 = -15
