from typing import List, Dict, Set, Tuple

import attrs

from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.code_changes import change
from databutler.datana.generic.corpus.code_changes.change_gen import (
    BaseCodeRemovalChangeGen,
)
from databutler.datana.viz.corpus import code_processors
from databutler.pat import astlib
from databutler.utils import code as codeutils

#  Name of the library, such as "pandas", "matplotlib" etc.
_LibraryNameT = str
#  Qualified function name
_QualFuncNameT = str
#  Unqualified function name
_UnqualFuncNameT = str
#  Name of the keyword argument.
_KwArgNameT = str
#  String containing code.
_CodeStringT = str


@attrs.define(eq=False, repr=False)
class VizMplConstKwArgRemover(BaseCodeRemovalChangeGen):
    eligible_fns: Set[Tuple[_LibraryNameT, _UnqualFuncNameT]]
    eligible_kw_names: Dict[Tuple[_LibraryNameT, _UnqualFuncNameT], Set[_KwArgNameT]]

    def _get_func_name_mappings(
        self, func: DatanaFunction
    ) -> Dict[_CodeStringT, _QualFuncNameT]:
        required_metadata_key = (
            code_processors.VizMplFuncNameExtractor.get_processor_metadata_key()
        )
        if func.metadata is None or required_metadata_key not in func.metadata:
            raise ValueError(
                f"Required metadata {required_metadata_key} not found in given Datana function. "
                f"Did you run {code_processors.VizMplFuncNameExtractor.__name__} on this function?"
            )

        fn_map_key = (
            code_processors.VizMplFuncNameExtractor.get_func_name_mappings_key()
        )
        func_name_mappings: Dict[_CodeStringT, _QualFuncNameT] = func.metadata[
            required_metadata_key
        ][fn_map_key]
        return func_name_mappings

    def _get_optional_args_mapping(
        self, func: DatanaFunction
    ) -> Dict[_CodeStringT, Set[_KwArgNameT]]:
        required_metadata_key = (
            code_processors.VizMplKeywordArgNormalizer.get_processor_metadata_key()
        )
        if func.metadata is None or required_metadata_key not in func.metadata:
            raise ValueError(
                f"Required metadata {required_metadata_key} not found in given Datana function. "
                f"Did you run {code_processors.KeywordArgNormalizer.__name__} on this function?"
            )

        arg_info_key = (
            code_processors.VizMplKeywordArgNormalizer.get_arg_infos_metadata_key()
        )
        arg_infos: Dict[_CodeStringT, Dict[str, List[str]]] = func.metadata[
            required_metadata_key
        ][arg_info_key]

        result: Dict[_CodeStringT, Set[_KwArgNameT]] = {}

        for code_str, info_dict in arg_infos.items():
            assert "optional_args" in info_dict
            result[code_str] = set(info_dict["optional_args"])

        return result

    def gen_changes(
        self, func: DatanaFunction
    ) -> List[change.SimpleAstLibRemovalChange]:
        code = func.code_str
        code_ast = astlib.parse(code)
        result: List[change.SimpleAstLibRemovalChange] = []

        #  Check if the necessary metadata is available for this generator.
        #  Specifically, we need to be able to accurately identify the qualified function names.
        func_name_mappings: Dict[
            _CodeStringT, _QualFuncNameT
        ] = self._get_func_name_mappings(func)
        #  Along with the really optional keyword args.
        optional_arg_mappings: Dict[
            _CodeStringT, Set[_KwArgNameT]
        ] = self._get_optional_args_mapping(func)

        for call in (n for n in astlib.walk(code_ast) if isinstance(n, astlib.Call)):
            call_code_str = codeutils.unparse_astlib_ast(call)
            if call_code_str not in func_name_mappings:
                continue

            if call_code_str not in optional_arg_mappings:
                continue

            qual_name = func_name_mappings[call_code_str]
            lib_name = qual_name.split(".")[0]
            unqual_name = qual_name.split(".")[-1]
            if lib_name == unqual_name:
                #  Is not a library call.
                continue

            #  Check eligibility.
            #  We only use the library name and the unqualified name because the internal organization of a package
            #  may change.
            if (lib_name, unqual_name) not in self.eligible_fns:
                continue

            if (lib_name, unqual_name) not in self.eligible_kw_names:
                continue

            optional_args: Set[_KwArgNameT] = optional_arg_mappings[call_code_str]
            eligible_names = (
                self.eligible_kw_names[lib_name, unqual_name] & optional_args
            )
            rev_ref_dict = {
                v: k for k, v in change.SimpleAstLibNodeRef.get_refs(code_ast).items()
            }

            for arg in call.args:
                if (
                    astlib.is_keyword_arg(arg)
                    and astlib.is_constant(arg.value)
                    and arg.keyword.value in eligible_names
                ):
                    ref = rev_ref_dict[arg]
                    result.append(
                        change.SimpleAstLibRemovalChange(
                            node_refs=[ref],
                            children=[],
                        )
                    )

        return result
