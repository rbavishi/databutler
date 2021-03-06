import collections
import inspect
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, List, Any

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import (
    DatanaFunctionProcessor,
)
from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import (
    CallDecoratorsGenerator,
    CallDecorator,
    Instrumentation,
    Instrumenter,
)
from databutler.pat.utils import miscutils
from databutler.utils import inspection, code as codeutils


@attrs.define(eq=False, repr=False)
class _KeywordArgsFinder(CallDecoratorsGenerator):
    """
    Instrumentation generator for intercepting function calls and tracking positional and optional parameters.
    """

    label_mappings: Dict[astlib.Call, List[str]] = attrs.Factory(dict)
    args_info_mappings: Dict[astlib.Call, Dict] = attrs.Factory(dict)

    def gen_decorators(
        self, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[
            astlib.BaseExpression, List[CallDecorator]
        ] = collections.defaultdict(list)

        for n in self.iter_calls(ast_root):
            miscutils.merge_defaultdicts_list(
                decorators, self.gen_finder(n, ast_root=ast_root)
            )

        return decorators

    def gen_finder(
        self, call: astlib.Call, ast_root: astlib.AstNode
    ) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        def finder(func, args, kwargs):
            try:
                sig = inspect.signature(func)

            except (ValueError, TypeError):
                #  Could not obtain the signature. Ignore the function.
                pass

            else:
                num_pos_args = len(args)
                dummy_args = [f"my-arg-{idx}" for idx in range(0, num_pos_args)]
                binding = sig.bind(*dummy_args, **kwargs)
                rev_mapping = {
                    v: k
                    for k, v in binding.arguments.items()
                    if isinstance(v, str) and v in dummy_args
                }
                if not all(i in rev_mapping for i in dummy_args):
                    #  There must be starred args. Skip these functions.
                    return

                #  We do not want to make positional-only args keyword args
                positional_only_args = inspection.get_positional_only_args(sig=sig)
                pos_kw_labels = [
                    rev_mapping[i]
                    if rev_mapping[i] not in positional_only_args
                    else None
                    for i in dummy_args
                ]

                #  Everything before the last None also has to be None since we can't mix up positional arguments
                #  with keywords arguments per Python syntax.
                last_none_idx = -1
                for idx, val in enumerate(pos_kw_labels):
                    if val is None:
                        last_none_idx = idx

                if last_none_idx != -1:
                    for idx in range(0, last_none_idx):
                        pos_kw_labels[idx] = None

                self.label_mappings[call] = pos_kw_labels
                self.args_info_mappings[call] = {
                    "required_args": inspection.get_required_args(sig=sig),
                    "optional_args": inspection.get_optional_args(sig=sig),
                }

        return {
            call.func: [
                CallDecorator(
                    callable=finder, does_not_return=True, needs_return_value=False
                )
            ]
        }


def _convert_pos_args_to_kw_args(
    code_ast: astlib.AstNode, finder: _KeywordArgsFinder
) -> Dict:
    """
    Keyword-normalizes the provided code using the dynamic information obtained by the finder.
    """
    new_nodes = {}
    for old_node, labels in finder.label_mappings.items():
        old_args = list(old_node.args)
        new_args = []
        ctr = 0
        for arg in old_args:
            if arg.keyword is not None:
                new_args.append(arg)
            else:
                kw = labels[ctr]
                if kw is not None:
                    new_args.append(
                        astlib.with_changes(
                            arg, keyword=astlib.create_name_expr(labels[ctr])
                        )
                    )
                else:
                    #  We cannot make this a keyword argument.
                    new_args.append(arg)

                ctr += 1

        new_node = astlib.with_changes(old_node, args=new_args)
        new_nodes[old_node] = new_node

    arg_info_dict = {}

    for node, info in finder.args_info_mappings.items():
        node = new_nodes.get(node, node)
        arg_info_dict[codeutils.unparse_astlib_ast(node)] = info

    modified_ast = astlib.with_deep_replacements(code_ast, new_nodes)

    return {
        "code": codeutils.unparse_astlib_ast(modified_ast),
        "arg_infos": arg_info_dict,
    }


@attrs.define(eq=False, repr=False)
class KeywordArgNormalizer(DatanaFunctionProcessor, ABC):
    """
    A processor that keyword-normalizes code. Keyword-normalization is converting every positional argument into
    a keyword argument. This implementation relies on using instrumentation to dynamically figure out the positional
    argument names of the functions used, and then modifies the code accordingly.
    """

    def _process(self, d_func: DatanaFunction) -> DatanaFunction:
        code = d_func.code_str
        #  Set up instrumentation.
        finder = _KeywordArgsFinder()

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
        self._run_function_code(
            func_code=inst_code,
            func_name=d_func.func_name,
            pos_args=d_func.get_pos_args() or [],
            kw_args=d_func.get_kw_args() or {},
            global_ctx=global_ctx,
        )

        #  Get the normalized code
        res = _convert_pos_args_to_kw_args(code_ast, finder)
        norm_code = res["code"]
        norm_metadata = {  # Additional metadata we would like to store in the result.
            self.get_arg_infos_metadata_key(): res["arg_infos"]
        }

        #  Assemble the result
        new_d_func = d_func.copy()
        new_d_func.code_str = norm_code
        new_d_func.metadata = new_d_func.metadata or {}
        new_d_func.metadata[self.get_processor_metadata_key()] = {
            self.get_old_code_metadata_key(): d_func.code_str,
            **norm_metadata,
        }

        return new_d_func

    @classmethod
    def get_processor_name(cls) -> str:
        return "keyword-arg-normalizer"

    @classmethod
    def get_old_code_metadata_key(cls) -> str:
        return "old_code"

    @classmethod
    def get_arg_infos_metadata_key(cls) -> str:
        return "arg_infos"

    @abstractmethod
    def _run_function_code(
        self,
        func_code: str,
        func_name: str,
        pos_args: List[Any],
        kw_args: Dict[str, Any],
        global_ctx: Dict[str, Any],
    ) -> Any:
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
