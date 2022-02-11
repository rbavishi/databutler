import difflib
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class CodeChangeToNatLangTask:
    few_shot_examples: List[few_shot.FewShotExampleCodeChangeAndNL]
    target_old_code: str
    target_new_code: str
    task_description: Optional[str] = None


@attrs.define(eq=False)
class BaseCodeChangeToNatLang(ABC):
    @abstractmethod
    def get_nl(self, task: CodeChangeToNatLangTask, num_results: int = 1) -> List[str]:
        """
        Generates natural language descriptions of the given code change with language-models.

        This must be implemented by all subclasses.

        :param task: A code-change-to-natural-language task.
        :param num_results: An integer representing the number of NL descriptions to generate. Note that the number
                            of descriptions actually returned *may be less* than `num_results`. This can happen if the
                            language model does not come up with enough unique descriptions.

        :return: A list of strings of size <= `num_results` corresponding to generated candidate description.
        """

    def get_nl_bullets(self, task: CodeChangeToNatLangTask) -> Iterator[str]:
        """
        Generates natural language description as a sequence of bullet points. This method should return an iterator.
        This helps the client consume as much as they need. However, the iterator will stop as soon as the model
        starts repeating itself.

        :param task: A code-change-to-natural-language task.

        :return: An iterator of strings where each string corresponds to a single bullet point description.
        """
        #  By default, just return a single bullet that is just the plain NL.
        yield self.get_nl(task, num_results=1)[0]
        return


@attrs.define(eq=False)
class FullCodeChangeToNatLang(BaseCodeChangeToNatLang):
    """
    A simple strategy that presents the old code and the new code and asks the model to come up with a description
    of the change. This is the most demanding of the model as it is responsible for identifying and localizing the
    change.
    """
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 256

    default_task_description: str = (
        "Describe the difference between the old Python code and new Python code snippets below."
    )

    def _get_diff_representation(self, old_code: str, new_code: str) -> List[str]:
        prompt_strs: List[str] = [f"Old Code:\n{old_code}\n",
                                  f"New Code:\n{new_code}\n"]

        return prompt_strs

    def _create_completion_prompt(self, task: CodeChangeToNatLangTask,
                                  generated_bullets: Optional[List[str]] = None) -> str:
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
        desc = self.default_task_description
        if task.task_description is not None:
            desc = task.task_description

        prompt_strs.append(desc)

        for ex in task.few_shot_examples:
            #  Add the old and new code as is.
            prompt_strs.extend(self._get_diff_representation(ex.old_code, ex.new_code))

            if isinstance(ex.nl, list):
                ex_nl_str = "\n".join(f"* {i}" for i in ex.nl)
                is_bullet.append(True)
                prompt_strs.append(f"Change Description:\n{ex_nl_str}\n")

            else:
                ex_nl_str = ex.nl
                is_bullet.append(False)
                #  Do not use a new-line if it is not a bulleted list.
                prompt_strs.append(f"Change Description: {ex_nl_str}\n")

            prompt_strs.append("----\n")

        if not (all(i for i in is_bullet) or all(not i for i in is_bullet)):
            raise ValueError("Few-shot examples must not mix the single-line description and the bullet-point format")

        if (not all(is_bullet)) and generated_bullets is not None:
            raise ValueError("Cannot supply generated bullets for single-line description prompts.")

        prompt_strs.extend(self._get_diff_representation(task.target_old_code,
                                                         task.target_new_code))

        if all(is_bullet):
            #  In either case, end with a '*' so the model knows it is starting the next bullet point.
            if generated_bullets is not None and len(generated_bullets) > 0:
                nl_str = "\n".join(f"* {i}" for i in generated_bullets)
                prompt_strs.append(f"Change Description:\n{nl_str}\n*")
            else:
                prompt_strs.append(f"Change Description:\n*")

        else:
            #  We do not use a * in the single-line description format.
            prompt_strs.append(f"Change Description:")

        return "\n".join(prompt_strs)

    def get_nl(self, task: CodeChangeToNatLangTask, num_results: int = 1) -> List[str]:
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

    def get_nl_bullets(self, task: CodeChangeToNatLangTask) -> Iterator[str]:
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


@attrs.define(eq=False)
class DiffToNatLang(FullCodeChangeToNatLang):
    """
    A strategy where instead of providing the complete new code, we only provide the diff to the model.
    Thus, this makes it easier for the model to look at what actually changed, so results should be better for
    this strategy in general.
    """
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 256

    default_task_description: str = (
        "Describe the given diff for the Python code snippets below."
    )

    def _get_simplified_diff(self, old_code: str, new_code: str) -> str:
        """
        Returns a simplified diff between the old code and the new code.

        Can be overriden by sub-classes to tailor the diff format used in the prompt.

        Args:
            old_code: A string for the old code.
            new_code: A string for the new code.

        Returns:
            A diff between old_code and new_code as a simplified unified diff.

        """
        #  We simplify a unified diff, by using no context and removing metadata about files etc.
        #  The output only contains '-' and '+'es, something like below:
        #  - c = call(var=a)
        #  + c = call(var=a, another=arg)
        diff_lines = list(difflib.unified_diff(old_code.split('\n'), new_code.split('\n'), n=0, lineterm=''))
        return "\n".join(line for line in diff_lines
                         if not (line.startswith("---") or line.startswith("+++") or line.startswith("@@")))

    def _get_diff_representation(self, old_code: str, new_code: str) -> List[str]:
        #  Return the diff instead of Old Code: and New Code: as done in FullCodeChangeToNatLang.
        prompt_strs: List[str] = [f"Old Code:\n{old_code}\n", "Diff:"]
        prompt_strs.extend(self._get_simplified_diff(old_code, new_code).split("\n"))
        prompt_strs.append("")

        return prompt_strs
