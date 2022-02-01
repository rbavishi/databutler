import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessor
from databutler.utils import code as codeutils


class CodeNormalizer(DatanaFunctionProcessor):
    def _process(self, d_func: DatanaFunction) -> DatanaFunction:
        #  Simply normalize the code and return the new function
        new_d_func = d_func.copy()
        new_d_func.code_str = codeutils.normalize_code(d_func.code_str)

        return new_d_func

    @classmethod
    def get_processor_name(cls) -> str:
        return "code-normalizer"
