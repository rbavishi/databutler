from abc import ABC, abstractmethod
from typing import List, Optional, Union, Iterator

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class CodeToNatLangTask:
    #  Few-shot examples to use for LM completion.
    few_shot_examples: List[few_shot.FewShotExampleCodeAndNL]
    #  The target code to explain.
    target_code: str
    #  An optional description of the task. LM performance generally goes up if a good description is provided.
    task_description: Optional[str] = None


@attrs.define(eq=False)
class BaseCodeToNatLang(ABC):
    @abstractmethod
    def get_nl(self, task: CodeToNatLangTask, num_results: int = 1) -> List[str]:
        """
        Generates natural language descriptions of code with language-models using the provided few-shot examples.

        This must be implemented by all subclasses.

        :param task: A code-to-natural-language task.
        :param num_results: An integer representing the number of NL descriptions to generate. Note that the number
                            of descriptions actually returned *may be less* than `num_results`. This can happen if the
                            language model does not come up with enough unique descriptions.

        :return: A list of strings of size <= `num_results` corresponding to generated candidate description.
        """

    def get_nl_bullets(self, task: CodeToNatLangTask) -> Iterator[str]:
        """
        Generates natural language description as a sequence of bullet points. This method should return an iterator.
        This helps the client consume as much as they need. However, the iterator will stop as soon as the model
        starts repeating itself.

        :param task: A code-to-natural-language task.

        :return: An iterator of strings where each string corresponds to a single bullet point description.
        """
        #  By default, just return a single bullet that is just the plain NL.
        yield self.get_nl(task, num_results=1)[0]
        return


@attrs.define(eq=False)
class SimpleCodeToNatLang(BaseCodeToNatLang):
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 256

    def _create_completion_prompt(self, task: CodeToNatLangTask, generated_bullets: Optional[List[str]] = None) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the beginning of the prompt.

        :param task: A code-to-natural-language task.
        :param generated_bullets: An optional list of strings corresponding to the NL bullets already generated.
                                  This is useful for asking the model to provide a completion in the context of what it
                                  has already output.

        :return: A string corresponding to the prompt to use for OpenAI completion.
        """
        prompt_strs: List[str] = []
        is_bullet = []

        # Adding the additional target description to the prompt strings
        if task.task_description is not None:
            prompt_strs.append(task.task_description)

        for ex in task.few_shot_examples:
            prompt_strs.append(f"Python Code:\n{ex.code}")
            if isinstance(ex.nl, list):
                ex_nl_str = "\n".join(f"* {i}" for i in ex.nl)
                is_bullet.append(True)
                prompt_strs.append(f"\nDescription:\n{ex_nl_str}\n")

            else:
                ex_nl_str = ex.nl
                is_bullet.append(False)
                #  Do not use a new-line if it is not a bulleted list.
                prompt_strs.append(f"\nDescription: {ex_nl_str}\n")

            prompt_strs.append("----")

        if not (all(i for i in is_bullet) or all(not i for i in is_bullet)):
            raise ValueError("Few-shot examples must not mix the single-line description and the bullet-point format")

        if (not all(is_bullet)) and generated_bullets is not None:
            raise ValueError("Cannot supply generated bullets for single-line description prompts.")

        prompt_strs.append(f"Python Code:\n{task.target_code}")

        if all(is_bullet):
            #  In either case, end with a '*' so the model knows it is starting the next bullet point.
            if generated_bullets is not None and len(generated_bullets) > 0:
                nl_str = "\n".join(f"* {i}" for i in generated_bullets)
                prompt_strs.append(f"\nDescription:\n{nl_str}\n*")
            else:
                prompt_strs.append(f"\nDescription:\n*")

        else:
            #  We do not use a * in the single-line description format.
            prompt_strs.append(f"\nDescription:")

        return "\n".join(prompt_strs)

    def get_nl(self, task: CodeToNatLangTask, num_results: int = 1) -> List[str]:
        """
        Creates a simple prompt stringing examples together and uses it to generate the descriptions.

        See base method for a description of the arguments and return value.
        """
        #  Ensure that the few-shot examples do not use bullet-points.
        if any(isinstance(ex.nl, list) for ex in task.few_shot_examples):
            raise ValueError("Few-shot examples cannot contain bullet-point descriptions "
                             "when generating single-line descriptions.")

        completion_prompt = self._create_completion_prompt(task)

        resp = langmodels.openai_completion(
            engine=self.engine,
            prompt=completion_prompt,
            temperature=self.temperature,
            num_completions=num_results,
            max_tokens=self.max_tokens,
            stop=["\n"],  # Use new-line as the stop-token for single-line descriptions.
            retry_wait_duration=60,
            max_retries=5,
            return_logprobs=False,
        )

        descriptions = list(set(
            c.text.strip() for c in resp.completions
        ))

        return descriptions

    def get_nl_bullets(self, task: CodeToNatLangTask) -> Iterator[str]:
        """
        Simply invokes the model as long as it produces a new bullet-point.

        See base method for a description of the arguments and return value.
        """

        #  Ensure that the few-shot examples all use bullet-points.
        if any(not isinstance(ex.nl, list) for ex in task.few_shot_examples):
            raise ValueError("Few-shot examples cannot contain single-line descriptions "
                             "when generating bullet-point descriptions.")

        generated_bullets: List[str] = []

        while True:
            #  We will keep asking the model for new points as long as the consumer of our iterator wants a new point,
            #  or until the model repeats itself.
            completion_prompt = self._create_completion_prompt(task, generated_bullets)

            resp = langmodels.openai_completion(
                engine=self.engine,
                prompt=completion_prompt,
                temperature=self.temperature,
                num_completions=1,
                max_tokens=self.max_tokens,
                stop=["\n"],  # Use new-line as the stop-token for single-line descriptions.
                retry_wait_duration=60,
                max_retries=5,
                return_logprobs=False,
            )

            new_bullet = resp.completions[0].text.strip()
            if new_bullet in generated_bullets:
                break

            generated_bullets.append(new_bullet)
            yield new_bullet
