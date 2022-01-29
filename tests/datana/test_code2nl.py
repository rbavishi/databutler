import itertools
import unittest

from databutler.datana.training import code2nl, few_shot, nl2code
from databutler.datana.utils import vizutils


class Code2NLTests(unittest.TestCase):
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

    def test_simple_2(self):
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

    def test_vizfunc_1(self):
        """
        Tests to generate basic visualization functions.
        """
        import numpy as np
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl = [
                    "Defines a function f that accepts a dataframe df, the column name x, and the column name y",
                    "Plot the y column of the dataframe against the x column"
                ],
                code = (
                    "import matplotlib.pyplot as plt\n"
                    "def f(df, x, y):\n"
                    "   return plt.plot(df[x], df[y])"
                )
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl = [
                    "Defines a function f that accepts a dataframe df, the column name x and the column name y, and a list of labels",
                    "Create a stacked plot the y column against the x column, with the labels as the stacks",
                ],
                code = (
                    "import matplotlib.pyplot as plt\n"
                    "def f(df, x, y, labels):\n"
                    "   plt.legend(loc='upper left')\n"
                    "   return plt.stackplot(df[x], df[y], labels=labels)"
                )
            )
        ]

        target_code = (
            "import matplotlib.pyplot as plt\n"
            "def f(df, col):\n"
            "    return plt.hist(df[col])\n"
        )

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to create a histogram..."
        generated_nl = list(itertools.islice(nl_generator.get_nl_bullets(few_shot_examples, target_code), 2))
        self.assertIsInstance(generated_nl, list)
        self.assertEqual(2, len(generated_nl))

        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        # Setup for inputs to the code
        df = {'input': [1, 2, 3, 4, 4, 5, 5, 5]}

        # Run the original code and save the plot
        ctx = {}
        exec(target_code, ctx)
        self.assertIn('f', ctx.keys())
        target_value, target_bins, _ = ctx['f'](df, 'input')

        # Run the regenerated code and save the plot
        ctx = {}
        exec(regenerated_code, ctx)
        self.assertIn('f', ctx.keys())
        generated_value, generated_bins, _ = ctx['f'](df, 'input')

        # Compare the two
        # Checks if the values of the histogram bins are equal
        self.assertTrue(np.array_equal(target_value, generated_value))
        # Checks if the edges of the bins are equal
        self.assertTrue(np.array_equal(target_bins, generated_bins))

    # Same as test_vizfunc_1 but instead uses image bytestream comparison to check equality.
    def test_vizfunc_2(self):
        """
        Tests to generate basic visualization functions.
        """
        import numpy as np
        few_shot_examples = [
            few_shot.FewShotExampleCodeAndNL(
                nl = [
                    "Defines a function f that accepts a dataframe df, the column name x, and the column name y",
                    "Plot the y column of the dataframe against the x column"
                ],
                code = (
                    "import matplotlib.pyplot as plt\n"
                    "def f(df, x, y):\n"
                    "   plt.plot(df[x], df[y])"
                )
            ),
            few_shot.FewShotExampleCodeAndNL(
                nl = [
                    "Defines a function f that accepts a dataframe df, the column name x and the column name y, and a list of labels",
                    "Create a stacked plot the y column against the x column, with the labels as the stacks",
                ],
                code = (
                    "import matplotlib.pyplot as plt\n"
                    "def f(df, x, y, labels):\n"
                    "   plt.legend(loc='upper left')\n"
                    "   plt.stackplot(df[x], df[y], labels=labels)"
                )
            )
        ]

        target_code = (
            "import matplotlib.pyplot as plt\n"
            "def f(df, col):\n"
            "   plt.hist(df[col])"
        )

        nl_generator = code2nl.SimpleCodeToNatLang()
        code_generator = nl2code.SimpleNatLangToCode()

        #  The generated NL should be something like "A function to create a histogram..."
        generated_nl = list(itertools.islice(nl_generator.get_nl_bullets(few_shot_examples, target_code), 2))
        self.assertIsInstance(generated_nl, list)
        self.assertEqual(2, len(generated_nl))

        regenerated_code = code_generator.get_code(few_shot_examples, generated_nl)

        # Setup for inputs to the code
        df = {'input': [1, 2, 3, 4, 4, 5, 5, 5]}

        target_fig = vizutils.run_viz_code_matplotlib_mp(target_code, {'df': df, 'col': 'input'}, 'f')
        generated_fig = vizutils.run_viz_code_matplotlib_mp(regenerated_code, {'df': df, 'col': 'input'}, 'f')

        # Compare the two
        self.assertTrue(vizutils.serialize_fig(target_fig) == vizutils.serialize_fig(generated_fig))





