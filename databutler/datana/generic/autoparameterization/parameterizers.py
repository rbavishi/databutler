from typing import Optional, List, Tuple

import attrs

from databutler.datana.generic.autoparameterization.few_shot import (
    FewShotExampleParameterization,
)
from databutler.utils import langmodels, code as codeutils


@attrs.define(eq=False, repr=False)
class ParameterizationTask:
    few_shot_examples: List[FewShotExampleParameterization]
    target_nl: str
    target_code: str
    task_description: Optional[str] = None


@attrs.define(eq=False, repr=False)
class SimpleParameterizer:
    temperature: float = 0.0
    engine: str = "code-davinci-001"
    #  How many maximum additional tokens should be utilized by the parameterized nl and code combined?
    tokens_surplus: int = 96

    stop_token: str = "END"

    def _create_completion_prompt(self, task: ParameterizationTask) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the beginning of the prompt.

        :param task: A parameterization task instance.
        :return: A string corresponding to the prompt to use for OpenAI completion.
        """
        prompt_strs: List[str] = []

        if task.task_description is not None:
            prompt_strs.append(task.task_description)

        #  First add in the few-shot examples.
        for ex in task.few_shot_examples:
            prompt_strs.append(f"Original Description:\n{ex.nl}\n")
            prompt_strs.append(f"Original Code:\n{ex.code}\n")
            prompt_strs.append(f"Parameterized Description:\n{ex.param_nl}\n")
            prompt_strs.append(
                f"Parameterized Code:\n{ex.param_code}\n{self.stop_token}\n"
            )
            prompt_strs.append("----")

        #  Now add in the targets
        prompt_strs.append(f"Original Description:\n{task.target_nl}\n")
        prompt_strs.append(f"Original Code:\n{task.target_code}\n")
        prompt_strs.append(f"Parameterized Description:\n")

        return "\n".join(prompt_strs)

    def parameterize(self, task: ParameterizationTask) -> Tuple[str, str]:
        pass

    def parallel_parameterize(
        self,
        tasks: List[ParameterizationTask],
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[Optional[Tuple[str, str]]]:
        """Like parameterize, but handles multiple tasks in parallel"""
        completion_prompts = [self._create_completion_prompt(task) for task in tasks]

        max_tokens = max(
            (
                len(
                    langmodels.tokenize(task.target_nl, engine=self.engine)["token_ids"]
                )
                + len(
                    langmodels.tokenize(task.target_code, engine=self.engine)[
                        "token_ids"
                    ]
                )
            )
            + self.tokens_surplus
            for task in tasks
        )

        responses = langmodels.openai_completion(
            engine=self.engine,
            prompts=completion_prompts,
            temperature=self.temperature,
            num_completions=1,
            max_tokens=max_tokens,
            stop=[self.stop_token],
            key_manager=key_manager,
        )

        results: List[Optional[Tuple[str, str]]] = []

        for resp, task in zip(responses, tasks):
            text = resp.completions[0].text
            if text.count("Parameterized Code:") != 1:
                results.append(None)
                print("WTF2", text)
                continue

            desc, code = text.split("Parameterized Code:")
            desc = desc.strip()
            try:
                results.append((desc, codeutils.normalize_code_fast(code)))
            except:
                results.append(None)

        return results
