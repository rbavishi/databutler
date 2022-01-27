import unittest

from databutler.datana.training import nl2code


class NL2CodeTestsRecursive(unittest.TestCase):
    def test_1(self):
        few_shot_examples = [
            nl2code.FewShotExample(
                nl=[
                    "A function that takes a number as input",
                    "Returns that factorial of that number"
                ],
                code=(
                    "def f(n):\n"
                    "   if n == 1:\n"
                    "       return 1\n"
                    "   else:\n"
                    "       return n * f(n-1)"
                )
            ),
            nl2code.FewShotExample(
                nl=[
                    "A function that takes a number as input",
                    "Returns the sum of the digits of the number"
                ],
                code=(
                    "def f(n):\n"
                    "   if n < 10:\n"
                    "       return n\n"
                    "   else:\n"
                    "       return f(n // 10) + n % 10"
                )
            ),
        ]

        target_nl = [
            "A function that takes in a number as input",
            "Returns that term of the fibonacci sequence"
        ]
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        # Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)

        self.assertIn('f', ctx)

        # Edge Cases
        self.assertEqual(0, ctx['f'](0))
        self.assertEqual(1, ctx['f'](1))

        # General Cases
        self.assertEqual(5, ctx['f'](5))
        self.assertEqual(34, ctx['f'](9))
        self.assertEqual(4181, ctx['f'](19))

    def test_2(self):
        few_shot_examples = [
            nl2code.FewShotExample(
                nl=[
                    "A function that takes a number as input",
                    "Returns that factorial of that number"
                ],
                code=(
                    "def f(n):\n"
                    "   if n == 1:\n"
                    "       return 1\n"
                    "   else:\n"
                    "       return n * f(n-1)"
                )
            ),
            nl2code.FewShotExample(
                nl=[
                    "A function that takes a number as input",
                    "Returns the sum of the digits of the number"
                ],
                code=(
                    "def f(n):\n"
                    "   if n < 10:\n"
                    "       return n\n"
                    "   else:\n"
                    "       return f(n // 10) + n % 10"
                )
            ),
        ]

        target_nl = [
            "A function that takes in two non-negative numbers, n and m, as input",
            "Returns the number of different partitions of n using parts up to m"
        ]
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        # Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)

        self.assertIn('f', ctx)

        # Edge Cases
        self.assertEqual(1, ctx['f'](0, 5))
        self.assertEqual(0, ctx['f'](5, 0))

        # General Cases
        self.assertEqual(9, ctx['f'](6, 4))
        self.assertEqual(7, ctx['f'](5, 5))
        self.assertEqual(627, ctx['f'](20, 20))