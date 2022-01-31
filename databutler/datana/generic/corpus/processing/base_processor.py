from abc import ABC, abstractmethod

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction


@attrs.define(eq=False, repr=False)
class DatanaFunctionProcessor(ABC):
    @abstractmethod
    def _process(self, d_func: DatanaFunction) -> DatanaFunction:
        """

        Args:
            d_func:

        Returns:

        """

    @classmethod
    @abstractmethod
    def get_processor_name(cls) -> str:
        """
        Returns the name of the processor as a string.
        """

    def run(self, d_func: DatanaFunction) -> DatanaFunction:
        new_func = self._process(d_func)

        #  Validation check(s)
        if new_func.uid != d_func.uid:
            raise AssertionError("Function UID not same after processing.")

        #  Maintain a history of processors applied.
        if new_func.metadata is None:
            new_func.metadata = {}

        old_metadata = d_func.metadata or {}
        key = 'processors_applied'
        processor_history = list(old_metadata.get(key, []))
        processor_history.append(self.get_processor_name())
        new_func.metadata[key] = processor_history

        return new_func
