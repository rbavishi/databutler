import collections
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessor
from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import CallDecoratorsGenerator, CallDecorator, Instrumentation, \
    Instrumenter
from databutler.pat.utils import miscutils
from databutler.utils import inspection, code as codeutils


@attrs.define(eq=False, repr=False)
class _FuncNameFinder(CallDecoratorsGenerator):
    """
    Instrumentation generator for intercepting function calls and extracting qualified names whenever possible.
    """
    func_name_mappings: Dict[astlib.Call, str] = attrs.Factory(dict)

    def gen_decorators(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[astlib.BaseExpression, List[CallDecorator]] = collections.defaultdict(list)

        for n in self.iter_calls(ast_root):
            miscutils.merge_defaultdicts_list(decorators, self.gen_finder(n, ast_root=ast_root))

        return decorators

    def gen_finder(self,
                   call: astlib.Call,
                   ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        def finder(func, args, kwargs):
            qual_name = inspection.get_fully_qualified_name(func)
            if qual_name is not None:
                self.func_name_mappings[call] = qual_name

        return {call.func: [CallDecorator(callable=finder,
                                          does_not_return=True,
                                          needs_return_value=False)]}


def _get_func_name_metadata(finder: _FuncNameFinder) -> Dict[str, str]:
    """
    Maps individual function calls (in text form) to the qualified function name.

    Args:
        finder: A FuncNameFinder instrumentation generator instance that has already been executed.

    Returns:
        A dictionary mapping function calls (in text form) to the qualified name of the corresponding function.
    """
    func_name_mappings: Dict[str, str] = {}
    for node, name in finder.func_name_mappings.items():
        func_name_mappings[codeutils.unparse_astlib_ast(node)] = name

    return func_name_mappings


@attrs.define(eq=False, repr=False)
class FuncNameExtractor(DatanaFunctionProcessor, ABC):
    """
    A processor that adds metadata about the functions and their qualified names used in a Datana function
    """
    def _process(self, d_func: DatanaFunction) -> DatanaFunction:
        code = d_func.code_str
        #  Set up instrumentation.
        finder = _FuncNameFinder()

        normalizer_instrumentation = Instrumentation.from_generators(finder)
        instrumenter = Instrumenter(normalizer_instrumentation)

        #  We use PAT's wrappers for ASTs as it is more expressive and powerful.
        #  The instrumentation library also relies on it.
        code_ast = astlib.parse(code)
        inst_ast, global_ctx = instrumenter.process(code_ast)
        inst_code = astlib.to_code(inst_ast)

        #  Execute the code as per the client domain's requirements.
        #  Once the instrumented code is run, the finder should have populated its
        #  internal data-structures for us to use.
        self._run_function_code(func_code=inst_code, func_name=d_func.func_name,
                                pos_args=d_func.get_pos_args() or [],
                                kw_args=d_func.get_kw_args() or {},
                                global_ctx=global_ctx)

        #  Get the normalized code
        func_name_mappings = _get_func_name_metadata(finder)

        #  Assemble the result
        new_d_func = d_func.copy()
        new_d_func.metadata = new_d_func.metadata or {}
        new_d_func.metadata[self.get_processor_metadata_key()] = {
            self.get_func_name_mappings_key(): func_name_mappings
        }

        return new_d_func

    @classmethod
    def get_processor_name(cls) -> str:
        return "func-name-extractor"

    @classmethod
    def get_func_name_mappings_key(cls) -> str:
        return "func_name_mappings"

    @abstractmethod
    def _run_function_code(self, func_code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any],
                           global_ctx: Dict[str, Any]) -> Any:
        """
        Runs the provided function with the given args and global context.

        Must be implemented by every class implementing FuncNameExtractor.

        Args:
            func_code: A string corresponding to the function to be executed.
            func_name: The name of the function as a string.
            pos_args: A list of positional arguments to be provided to the function.
            kw_args: A dictionary of keyword arguments to be provided to the function.
            global_ctx: The global context in which to run the function.
        """
