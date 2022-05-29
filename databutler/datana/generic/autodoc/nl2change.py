import difflib
import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import attrs

from databutler.datana.generic.autodoc import few_shot
from databutler.utils import langmodels


@attrs.define(eq=False)
class NatLangToCodeChangeTask:
    few_shot_examples: List[few_shot.FewShotExampleCodeChangeAndNL]
    target_old_code: str
    target_nl: Union[str, List[str]]
    target_blanked: Optional[str] = None
    task_description: Optional[str] = None
    output_prefix: Optional[str] = None


@attrs.define(eq=False, repr=False)
class BaseNatLangToCodeChange(ABC):
    @abstractmethod
    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        """
        Generates new code given the old code and a natural language description using language-models.

        This must be implemented by all subclasses.

        Args:
            task: A nl-to-code-change task instance.

        Returns:
            A string corresponding to the generated code.
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

        Args:
            task: A nl-to-code-change task instance.

        Returns:
            A string corresponding to the prompt to use for OpenAI completion.
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
            retrieve_top_tokens=False,
        )

        text = resp.completions[0].text
        if task.output_prefix is not None:
            #  We need to add the output prefix back.
            #  Note we remove trailing whitespace in the prompt generation, so need to do the same thing here.
            text = f"{task.output_prefix.rstrip()}{text}"

        return text


class ModelFailedError(Exception):
    """
    An exception for when the model fails to do the right thing.
    """


@attrs.define(eq=False, repr=False)
class NatLangToStmtBlanks(BaseNatLangToCodeChange):
    """
    A change generation strategy where the model is instructed to fill in the statement level blanks in the new code.
    This spares the model from having to recreate or repeat the rest of the code verbatim.
    """
    temperature: float = 0.0
    engine: str = 'code-davinci-001'
    max_tokens: int = 512

    all_at_once: bool = True
    stop_token: str = "END"
    default_blank_word: str = "BLANK_STATEMENT"
    default_task_description: str = (
        f"Replace the blanks with Python code given the description of the change and the original code.\n"
    )

    @classmethod
    def create_blanks_and_answers(cls, old_code: str, new_code: str, blank_word: str) -> Tuple[str, List[str]]:
        old_lines = old_code.split("\n")
        new_lines = new_code.split("\n")

        n = max(len(old_lines), len(new_lines))
        groups = list(difflib.SequenceMatcher(isjunk=None, a=old_lines, b=new_lines).get_grouped_opcodes(n=n))
        assert len(groups) == 1

        new_lines_with_blanks: List[str] = []
        answers: List[str] = []
        #  See difflib.py to learn the inner workings of SequenceMatcher
        ctr = 1
        for tag, i1, i2, j1, j2 in groups[0]:
            if tag == 'equal':
                new_lines_with_blanks.extend(old_lines[i1: i2])
                continue

            if tag in {'replace', 'delete'}:
                #  These are not part of the new code, so skip.
                pass

            if tag in {'replace', 'insert'}:
                for line in new_lines[j1: j2]:
                    #  Get the leading whitespace and preserve it
                    leading = "".join(itertools.takewhile(str.isspace, line))
                    new_lines_with_blanks.append(f"{leading}{blank_word}-{ctr}")
                    ctr += 1
                    answers.append(line.strip())

        return "\n".join(new_lines_with_blanks), answers

    def _create_completion_prompt(self, task: NatLangToCodeChangeTask, blank_word: str,
                                  generated_blanks: Optional[List[str]] = None) -> str:
        """
        Helper method to create the prompt. Strings the few-shot examples together, and adds the target description to
        the beginning of the prompt.

        Args:
            task: A nl-to-code-change task instance.

        Returns:
            A string corresponding to the prompt to use for OpenAI completion.
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
            ex_blanked_code, ex_answers = self.create_blanks_and_answers(ex.old_code, ex.new_code, blank_word)
            prompt_strs.append(f"Blanked Code:\n{ex_blanked_code}\n")
            #  Put in the answers one-by-one
            prompt_strs.append(f"Answers:")
            if self.all_at_once:
                #  We will use the stop token at the end.
                for ans_idx, ans in enumerate(ex_answers, 1):
                    prompt_strs.append(f"{self.default_blank_word}-{ans_idx}: {ans}")

                prompt_strs.append(self.stop_token)

            else:
                #  We use the stop-token to signal the end of each blank.
                for ans_idx, ans in enumerate(ex_answers, 1):
                    prompt_strs.append(f"{self.default_blank_word}-{ans_idx}: {ans} {self.stop_token}")

            prompt_strs.append("\n----")

        #  Now add in the target old code.
        prompt_strs.append(f"\nOld Code:\n{task.target_old_code}\n")

        #  Next, add in the target natural language i.e. the NL describing the change from the old code.
        if isinstance(task.target_nl, list):
            #  If NL is in the form of bullet points, format accordingly.
            nl_str = "\n".join(f"* {i}" for i in task.target_nl)
        else:
            nl_str = task.target_nl

        prompt_strs.append(f"Change Description:\n{nl_str}\n")
        prompt_strs.append(f"Blanked Code:\n{task.target_blanked}\n")

        prompt_strs.append(f"Answers:")
        num_blanks = task.target_blanked.count(self.default_blank_word)
        if (not self.all_at_once) and len(generated_blanks) < num_blanks:
            for idx, ans in enumerate(generated_blanks, 1):
                prompt_strs.append(f"{self.default_blank_word}-{idx}: {ans} {self.stop_token}")

            prompt_strs.append(f"{self.default_blank_word}-{len(generated_blanks) + 1}:")

        return "\n".join(prompt_strs)

    def get_changed_code(self, task: NatLangToCodeChangeTask, blank_word: Optional[str] = None) -> str:
        if blank_word is None:
            blank_word = self.default_blank_word

        if task.target_blanked is None:
            raise ValueError(f"{self.__class__.__name__} requires `target_blank` to be supplied in the task.")

        num_blanks = task.target_blanked.count(blank_word)

        if self.all_at_once:
            completion_prompt = self._create_completion_prompt(task, blank_word)

            resp = langmodels.openai_completion(
                engine=self.engine,
                prompt=completion_prompt,
                temperature=self.temperature,
                num_completions=1,
                max_tokens=self.max_tokens,
                stop=[self.stop_token],
                retry_wait_duration=60,
                max_retries=5,
                retrieve_top_tokens=False,
            )

            text = resp.completions[0].text

            generated_blanks: List[str] = []
            for line in text.split('\n'):
                line = line.strip()
                if line == "":
                    continue

                if ":" not in line:
                    continue

                code = ":".join(line.split(':')[1:])
                generated_blanks.append(code)

            if len(generated_blanks) != num_blanks:
                raise ModelFailedError(f"Model did not fill in all the blanks successfully")

        else:
            generated_blanks: List[str] = []
            for _ in range(num_blanks):
                #  We ask for blanks one by one. This ensures that the model fills all the blanks, unlike the
                #  all-at-once case. This does increase the number of calls to the model.
                completion_prompt = self._create_completion_prompt(task, blank_word, generated_blanks=generated_blanks)

                resp = langmodels.openai_completion(
                    engine=self.engine,
                    prompt=completion_prompt,
                    temperature=self.temperature,
                    num_completions=1,
                    max_tokens=self.max_tokens,
                    stop=[self.stop_token],  # Use new-line as the stop-token for single-line descriptions.
                    retry_wait_duration=60,
                    max_retries=5,
                    retrieve_top_tokens=False,
                )

                generated_blanks.append(resp.completions[0].text.strip())

        final_code = task.target_blanked
        for idx, ans in enumerate(generated_blanks, 1):
            final_code = final_code.replace(f"{blank_word}-{idx}", ans)

        return final_code


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
