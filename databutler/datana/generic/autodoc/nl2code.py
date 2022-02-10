from abc import ABC, abstractmethod
from typing import List, Optional, Union

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class BaseNatLangToCode(ABC):
    @abstractmethod
    def get_code(self, few_shot_examples: List[few_shot.FewShotExampleCodeAndNL], target_nl: Union[str, List[str]],
                output_prefix: Optional[str] = None, task_desc: Optional[List[str]] = []) -> str:
        """
        Generates code with language-models using the provided few-shot examples.

        The output_prefix can be used to constrain and influence the output of the model by enforcing that the output
        begins with the supplied prefix.

        This must be implemented by all subclasses.

        :param few_shot_examples: A list of few_shot.FewShotExampleCodeAndNL instances.
        :param target_nl: A string or a list of strings (bullet points) describing the code to generate.
        :param output_prefix: A string corresponding to the prefix the generated code *must* start with.
        :return: A string corresponding to the generated code. If the output_prefix is supplied, it is included
                 in the output.
        """


@attrs.define(eq=False)
class SimpleNatLangToCode(BaseNatLangToCode):
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 512

    stop_token: str = "END"

    def _create_completion_prompt(self, few_shot_examples: List[few_shot.FewShotExampleCodeAndNL],
                                  target_nl: Union[str, List[str]], task_desc: List[str], output_prefix: Optional[str] = None) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the end of the prompt.

        :param few_shot_examples: A list of few_shot.FewShotExampleCodeAndNL instances.
        :param target_nl: A string or a list of strings (bullet points) describing the code to generate.
        :param output_prefix: A string corresponding to the prefix the generated code *must* start with.
        :return: A string corresponding to the prompt to use for OpenAI completion.
        """
        prompt_strs: List[str] = []

        prompt_strs.extend(task_desc)

        #  First add in the few-shot examples.
        for ex in few_shot_examples:
            if isinstance(ex.nl, list):
                #  If NL is in the form of bullet points, format accordingly.
                ex_nl_str = "\n".join(f"* {i}" for i in ex.nl)
            else:
                ex_nl_str = ex.nl

            prompt_strs.append(f"Description:\n{ex_nl_str}")
            #  Use a special stop-token to signal the end of the code.
            #  Note that this is better than using something like a new-line, because the latter will cause issues
            #  if we want to generate multi-line code.
            prompt_strs.append(f"\nPython Code:\n{ex.code}\n{self.stop_token}\n")
            prompt_strs.append("----")

        #  Now add in the target natural language i.e. the NL to convert to code.
        if isinstance(target_nl, list):
            #  If NL is in the form of bullet points, format accordingly.
            nl_str = "\n".join(f"* {i}" for i in target_nl)
        else:
            nl_str = target_nl

        prompt_strs.append(f"Description:\n{nl_str}")
        prompt_strs.append(f"\nPython Code:")
        if output_prefix is not None:
            prompt_strs.append(output_prefix.rstrip())  # OpenAI lang. models do not work well with trailing whitespace.

        return "\n".join(prompt_strs)

    def get_code(self, few_shot_examples: List[few_shot.FewShotExampleCodeAndNL], target_nl: Union[str, List[str]],
                output_prefix: Optional[str] = None, task_desc: Optional[List[str]] = []) -> str:
        """
        Creates a simple prompt stringing examples together and uses it to generate the code.

        See base method for a description of the arguments and return value.
        """
        task_description = task_desc if task_desc is not None else []
        completion_prompt = self._create_completion_prompt(few_shot_examples, target_nl, task_description, output_prefix)

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
        if output_prefix is not None:
            #  We need to add the output prefix back.
            #  Note we remove trailing whitespace in the prompt generation, so need to do the same thing here.
            text = f"{output_prefix.rstrip()}{text}"

        return text
