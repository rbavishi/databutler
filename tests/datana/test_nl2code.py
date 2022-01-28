import unittest

from databutler.datana.training import nl2code, few_shot
from tests.datana.utils.utils import read_file


class NL2CodeTests(unittest.TestCase):
    def test_simple_1(self):
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

        target_nl = "A function to subtract two numbers"
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertIn('f', ctx.keys())
        self.assertEqual(5, ctx['f'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['f'](5, 20))  # 5 - 20 = -15

        #  Try with output prefixes. Use specific variable names.
        output_prefix = "def func(x, y):\n"
        generated_code = generator.get_code(few_shot_examples, target_nl, output_prefix)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertIn('func', ctx.keys())
        self.assertEqual(5, ctx['func'](10, 5))  # 10 - 5 = 5
        self.assertEqual(-15, ctx['func'](5, 20))  # 5 - 20 = -15

    def test_simple_2(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "A function that takes two integers as input",
                    "Returns their xor"
                ],
                code=(
                    "def f(a, b):\n"
                    "    return a ^ b"
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

        target_nl = [
            "A function that takes two strings as input",
            "Returns their concatenation"
        ]
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertEqual("ab", ctx['f']("a", "b"))
        self.assertEqual("10", ctx['f']("1", "0"))

        #  Try with output prefixes. Use specific variable names.
        output_prefix = "def func(x, y):\n"
        generated_code = generator.get_code(few_shot_examples, target_nl, output_prefix)

        #  Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)
        self.assertEqual("ab", ctx['func']("a", "b"))
        self.assertEqual("10", ctx['func']("1", "0"))

    def test_recursive_1(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
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
            few_shot.FewShotExampleCodeAndNL(
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

    def test_recursive_2(self):
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
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
            few_shot.FewShotExampleCodeAndNL(
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

    def test_datastructs_1(self):
        """
        Tests creation of stack data structure with push and pop features.
        """
        test_files_loc = 'tests/datana/test_files/datastructs_'
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "Defines a class that implements the queue data structure"
                ],
                code= read_file(f'{test_files_loc}queue')
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl=[
                    "Defines a class that implements the priority queue data structure"
                ],
                code= read_file(f'{test_files_loc}prioqueue')
            ),
        ]

        target_nl = [
            "Defines a class that implements the stack data structure"
        ]
        generator = nl2code.SimpleNatLangToCode()
        generated_code = generator.get_code(few_shot_examples, target_nl)

        # Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)

        self.assertIn('Stack', ctx)
        stack = ctx['Stack']()

        # Data structure tests
        self.assertEqual(True, stack.isEmpty())
        stack.push(1)
        self.assertEqual(False, stack.isEmpty())
        self.assertEqual(1, stack.pop())
        self.assertEqual(True, stack.isEmpty())

    def test_datastructs_2(self):
        """
        Tests creation of priority queue data structure that accepts a priority function with push and pop features.
        """
        test_files_loc = 'tests/datana/test_files/datastructs_'
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl = ["Defines a class that implements the queue data structure"],
                code = read_file(f'{test_files_loc}queue')
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl = ["Defines a class that implements the stack data structure"],
                code = read_file(f'{test_files_loc}stack')
            ),
        ]

        target_nl = [
            "Defines a class that implements a priority queue data structure that accepts a priority function",
            "making all the necessary imports"
        ]
        generator = nl2code.SimpleNatLangToCode()

        generated_code = generator.get_code(few_shot_examples, target_nl)

        # Run the generated code to see if it does the right thing
        ctx = {}
        exec(generated_code, ctx)

        self.assertIn('PriorityQueue', ctx)
        queue = ctx['PriorityQueue'](lambda x: -x)

        # Data structure tests
        self.assertEqual(True, queue.isEmpty())
        queue.push(1)
        self.assertEqual(False, queue.isEmpty())
        queue.push(2)
        self.assertEqual(2, queue.pop())
        self.assertEqual(1, queue.pop())
        self.assertEqual(True, queue.isEmpty())
