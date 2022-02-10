from abc import ABC, abstractmethod
from typing import List, Optional, Union

import attrs

from databutler.datana.generic.autodoc import few_shot


@attrs.define(eq=False)
class NatLangToCodeChangeTask:
    few_shot_examples: List[few_shot.FewShotExampleCodeChangeAndNL]
    target_old_code: str
    target_nl: Union[str, List[str]]
    task_description: Optional[str] = None


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
