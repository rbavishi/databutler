import unittest

from databutler.utils import langmodels


class CompletionWrapperTests(unittest.TestCase):
    def test_simple_1(self):
        engine = "code-davinci-001"
        prompt = (
            "Q: Capital of France?\n"
            "A: Paris\n"
            "Q: Capital of Belgium?\n"
            "A:"
        )

        resp1 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop='\n',
        )

        resp2 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop='\n',
            return_logprobs=True
        )

        self.assertEqual("Brussels", resp1.completions[0].text.strip())
        self.assertEqual("Brussels", resp2.completions[0].text.strip())
        self.assertIsNone(resp1.completions[0].logprob)
        self.assertIsNotNone(resp2.completions[0].logprob)
