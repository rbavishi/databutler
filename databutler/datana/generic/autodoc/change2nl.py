from abc import ABC, abstractmethod
from typing import List, Optional, Iterator

import attrs

from databutler.datana.generic.autodoc import few_shot


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
