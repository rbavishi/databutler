import unittest

from databutler.datana.training import nl2code
from tests.datana.utils.utils import read_file


class NL2CodeTestsDataStructs(unittest.TestCase):
    def test_1(self):
        """
        Tests creation of stack data structure with push and pop features.
        """
        test_files_loc = 'tests/datana/test_files/datastructs_'
        few_shot_examples = [
            nl2code.FewShotExample(
                nl=[
                    "Defines a class that implements the queue data structure"
                ],
                code= read_file(f'{test_files_loc}queue')
            ),
            nl2code.FewShotExample(
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

    def test_2(self):
        """
        Tests creation of priority queue data structure that accepts a priority function with push and pop features.
        """
        test_files_loc = 'tests/datana/test_files/datastructs_'
        few_shot_examples = [
            nl2code.FewShotExample(
                nl = ["Defines a class that implements the queue data structure"],
                code = read_file(f'{test_files_loc}queue')
            ),
            nl2code.FewShotExample(
                nl = ["Defines a class that implements the stack data structure"],
                code = read_file(f'{test_files_loc}stack')
            ),
        ]

        target_nl = ["Defines a class that implements a priority queue data structure that accepts a priority function"]
        generator = nl2code.SimpleNatLangToCode()

        # TODO: Currently this prefix is necessary. Is there a way to make it not be so?
        output_prefix = "import heapq\n"
        generated_code = generator.get_code(few_shot_examples, target_nl, output_prefix)

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
