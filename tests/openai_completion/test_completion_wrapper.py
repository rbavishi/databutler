import unittest

from databutler.utils import langmodels


class CompletionWrapperTests(unittest.TestCase):
    def test_simple_1(self):
        engine = "code-davinci-002"
        prompt = "Q: Capital of France?\n" "A: Paris\n" "Q: Capital of Belgium?\n" "A:"

        resp1 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop="\n",
        )

        resp2 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop="\n",
            retrieve_top_tokens=True,
        )

        self.assertEqual("Brussels", resp1.completions[0].text.strip())
        self.assertEqual("Brussels", resp2.completions[0].text.strip())
        self.assertIsNone(resp1.completions[0].top_logprobs)
        self.assertIsNotNone(resp2.completions[0].top_logprobs)

    def test_simple_2(self):
        engine = "code-davinci-002"
        prompt = (
            "Q: Capital of France?\n" "A: Paris\n" "Q: Capital of Belgium?\n" "A: Bru"
        )

        resp1 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop="\n",
        )

        resp2 = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop="\n",
            retrieve_top_tokens=True,
        )

        self.assertEqual("ssels", resp1.completions[0].text.strip())
        self.assertEqual("ssels", resp2.completions[0].text.strip())
        self.assertIsNone(resp1.completions[0].top_logprobs)
        self.assertIsNotNone(resp2.completions[0].top_logprobs)
        self.assertEqual(1, len(resp2.completions[0].top_logprobs))

    def test_retrieve_top_tokens_hard_1(self):
        engine = "code-davinci-002"
        prompt = (
            "Q: Capital of France?\n"
            "A: Paris is the capital\n"
            "Q: Capital of Belgium?\n"
            "A: Bru"
        )

        resp = langmodels.openai_completion(
            engine=engine,
            prompt=prompt,
            max_tokens=5,
            stop="\n",
            retrieve_top_tokens=True,
        )

        self.assertEqual("ssels is the capital", resp.completions[0].text.strip())
        self.assertIsNotNone(resp.completions[0].top_logprobs)
        self.assertEqual(4, len(resp.completions[0].top_logprobs))
