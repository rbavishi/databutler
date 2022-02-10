from abc import ABC, abstractmethod
from typing import List, Optional, Union

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class NatLangToCodeChangeTask:
    few_shot_examples: List[few_shot.FewShotExampleCodeChangeAndNL]
    target_old_code: str
    target_nl: Union[str, List[str]]
    task_description: Optional[str] = None
    output_prefix: Optional[str] = None


@attrs.define(eq=False, repr=False)
class BaseNatLangToCodeChange(ABC):
    @abstractmethod
    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        """
        Generates new code given the old code and a natural language description using language-models.

        This must be implemented by all subclasses.

        :param task: A nl-to-code-change task instance.
        :return: A string corresponding to the generated code.
        """


@attrs.define(eq=False, repr=False)
class NatLangToNewCode(BaseNatLangToCodeChange):
    """
    A change generation strategy where the model is instructed to generated the new code in its entirety.
    This is the most demanding strategy of the model as it needs to get all the aspects right.
    """
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 512

    stop_token: str = "END"
    default_task_description: str = "Generate the new code from the old code given the description of the change.\n"

    def _create_completion_prompt(self, task: NatLangToCodeChangeTask) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the beginning of the prompt.

        :param task: A nl-to-code-change task instance.
        :return: A string corresponding to the prompt to use for OpenAI completion.
        """
        prompt_strs: List[str] = []

        desc = self.default_task_description
        if task.task_description is not None:
            desc = task.task_description

        prompt_strs.append(desc)

        #  First add in the few-shot examples.
        for ex in task.few_shot_examples:
            #  First, the old code.
            prompt_strs.append(f"\nOld Code:\n{ex.old_code}\n")

            #  Next, the NL, or the description of the change.
            if isinstance(ex.nl, list):
                #  If NL is in the form of bullet points, format accordingly.
                ex_nl_str = "\n".join(f"* {i}" for i in ex.nl)
            else:
                ex_nl_str = ex.nl

            prompt_strs.append(f"Change Description:\n{ex_nl_str}\n")
            #  Use a special stop-token to signal the end of the code.
            #  Note that this is better than using something like a new-line, because the latter will cause issues
            #  if we want to generate multi-line code.
            prompt_strs.append(f"New Code:\n{ex.new_code}\n{self.stop_token}\n")
            prompt_strs.append("----")

        #  Now add in the target old code.
        prompt_strs.append(f"\nOld Code:\n{task.target_old_code}\n")

        #  Next, add in the target natural language i.e. the NL describing the change from the old code.
        if isinstance(task.target_nl, list):
            #  If NL is in the form of bullet points, format accordingly.
            nl_str = "\n".join(f"* {i}" for i in task.target_nl)
        else:
            nl_str = task.target_nl

        prompt_strs.append(f"Change Description:\n{nl_str}\n")
        prompt_strs.append(f"New Code:")
        if task.output_prefix is not None:
            #  OpenAI lang. models do not work well with trailing whitespace.
            prompt_strs.append(task.output_prefix.rstrip())

        return "\n".join(prompt_strs)

    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        """
        Creates a simple prompt stringing examples together and uses it to generate the code.

        See base method for a description of the arguments and return value.
        """
        completion_prompt = self._create_completion_prompt(task)

        resp = langmodels.openai_completion(
            engine=self.engine,
            prompt=completion_prompt,
            temperature=self.temperature,
            num_completions=1,
            max_tokens=self.max_tokens,
            stop=[self.stop_token],
            retry_wait_duration=60,
            max_retries=5,
            return_logprobs=False,
        )

        text = resp.completions[0].text
        if task.output_prefix is not None:
            #  We need to add the output prefix back.
            #  Note we remove trailing whitespace in the prompt generation, so need to do the same thing here.
            text = f"{task.output_prefix.rstrip()}{text}"

        return text


@attrs.define(eq=False, repr=False)
class NatLangToDiff(BaseNatLangToCodeChange):
    """
    A change generation strategy where the model is instructed to simply generate the statement-level diff from the
    original code. This spares the model from having to recreate or repeat the rest of the code verbatim.
    """

    def _create_completion_prompt(self, task: NatLangToCodeChangeTask) -> str:
        pass

    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        pass


@attrs.define(eq=False, repr=False)
class NatLangToFITB(BaseNatLangToCodeChange):
    """
    A change generation strategy where the model is instructed to fill in the blank at the specified position.
    This can be easier for the model than both NatLangToNewCode and NatLangToDiff as the localization problem is solved
    at a very fine-grained level for the model.
    """

    def _create_completion_prompt(self, task: NatLangToCodeChangeTask) -> str:
        pass

    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        pass
