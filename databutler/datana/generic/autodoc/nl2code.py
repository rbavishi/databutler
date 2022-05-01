from abc import ABC, abstractmethod
from typing import List, Optional, Union

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class NatLangToCodeTask:
    #  Few-shot examples to use for LM completion.
    few_shot_examples: List[few_shot.FewShotExampleCodeAndNL]
    #  A string or a list of strings (bullet points) describing the code to generate.
    target_nl: Union[str, List[str]]
    #  An optional description of the task. LM performance generally goes up if a good description is provided.
    task_description: Optional[str] = None
    #  A string corresponding to the prefix the generated code *must* start with. This can be utilized to constrain
    #  and influence the output of the language model.
    output_prefix: Optional[str] = None


@attrs.define(eq=False)
class BaseNatLangToCode(ABC):
    @abstractmethod
    def get_code(self, task: NatLangToCodeTask, *args, **kwargs) -> str:
        """
        Generates code with language-models using the provided few-shot examples.

        The output_prefix can be used to constrain and influence the output of the model by enforcing that the output
        begins with the supplied prefix.

        This must be implemented by all subclasses.

        :param task: A nl-to-code task instance.
        :return: A string corresponding to the generated code. If the output_prefix is supplied in the task,
            it is included in the output.
        """


@attrs.define(eq=False)
class SimpleNatLangToCode(BaseNatLangToCode):
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 512

    stop_token: str = "END"
    _stop_token_id: Optional[int] = None
    _newline_token_id: Optional[int] = None

    def _create_completion_prompt(self, task: NatLangToCodeTask) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the beginning of the prompt.

        :param task: A nl-to-code task instance.
        :return: A string corresponding to the prompt to use for OpenAI completion.
        """
        prompt_strs: List[str] = []

        if task.task_description is not None:
            prompt_strs.append(task.task_description)

        #  First add in the few-shot examples.
        for ex in task.few_shot_examples:
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
        if isinstance(task.target_nl, list):
            #  If NL is in the form of bullet points, format accordingly.
            nl_str = "\n".join(f"* {i}" for i in task.target_nl)
        else:
            nl_str = task.target_nl

        prompt_strs.append(f"Description:\n{nl_str}")
        prompt_strs.append(f"\nPython Code:")
        if task.output_prefix is not None:
            #  OpenAI lang. models do not work well with trailing whitespace.
            prompt_strs.append(task.output_prefix.rstrip())

        return "\n".join(prompt_strs)

    def _get_stop_token_id(self) -> int:
        if self._stop_token_id is None:
            print(langmodels.tokenize(self.stop_token, engine=self.engine))
            self._stop_token_id = langmodels.tokenize(self.stop_token, engine=self.engine)['token_ids'][0]

        return self._stop_token_id

    def _get_newline_token_id(self) -> int:
        if self._newline_token_id is None:
            self._newline_token_id = langmodels.tokenize("\n", engine=self.engine)['token_ids'][0]

        return self._newline_token_id

    def get_code(self, task: NatLangToCodeTask,
                 allowed_tokens: Optional[Union[str, List[int]]] = None,
                 allowed_tokens_bias: int = 100) -> str:
        """
        Creates a simple prompt stringing examples together and uses it to generate the code.

        See base method for a description of the arguments and return value.
        """
        completion_prompt = self._create_completion_prompt(task)

        logit_bias = {}
        max_tokens = self.max_tokens
        if allowed_tokens is not None:
            if isinstance(allowed_tokens, str):
                allowed_token_ids = langmodels.tokenize(allowed_tokens, engine=self.engine)['token_ids']
                max_tokens = len(allowed_token_ids) + 64
            else:
                allowed_token_ids = allowed_tokens

            logit_bias = {str(i): allowed_tokens_bias for i in allowed_token_ids}
            logit_bias[str(self._get_stop_token_id())] = allowed_tokens_bias
            logit_bias[self._get_newline_token_id()] = allowed_tokens_bias

        resp = langmodels.openai_completion(
            engine=self.engine,
            prompt=completion_prompt,
            temperature=self.temperature,
            num_completions=1,
            max_tokens=max_tokens,
            stop=[self.stop_token],
            retry_wait_duration=60,
            max_retries=5,
            return_logprobs=False,
            logit_bias=logit_bias,
        )

        text = resp.completions[0].text
        if task.output_prefix is not None:
            #  We need to add the output prefix back.
            #  Note we remove trailing whitespace in the prompt generation, so need to do the same thing here.
            text = f"{task.output_prefix.rstrip()}{text}"

        return text
