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


@attrs.define(eq=False, repr=False)
class NatLangToNewCode(BaseNatLangToCodeChange):
    """
    A change generation strategy where the model is instructed to generated the new code in its entirety.
    This is the most demanding strategy of the model as it needs to get all the aspects right.
    """

    def _create_completion_prompt(self, task: NatLangToCodeChangeTask) -> str:
        pass

    def get_changed_code(self, task: NatLangToCodeChangeTask) -> str:
        pass


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
