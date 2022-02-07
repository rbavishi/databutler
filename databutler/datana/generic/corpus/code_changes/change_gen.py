from abc import ABC, abstractmethod
from typing import List

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.code_changes.change import BaseCodeChange, BaseCodeRemovalChange


@attrs.define(eq=False, repr=False)
class BaseCodeChangeGen(ABC):
    """
    Base class for all code-change generators for datana functions.
    """
    @abstractmethod
    def gen_changes(self, func: DatanaFunction) -> List[BaseCodeChange]:
        """
        Generates the possible code-changes for the code in the provided Datana function.

        Args:
            func: A datana function for which code changes need to be generate

        Returns:
            A list of code-change instances for the code corresponding to the Datana function.
        """


class BaseCodeRemovalChangeGen(BaseCodeChangeGen, ABC):
    """
    Base class for all code-removal based change generators for datana functions.
    """
    @abstractmethod
    def gen_changes(self, func: DatanaFunction) -> List[BaseCodeRemovalChange]:
        pass
