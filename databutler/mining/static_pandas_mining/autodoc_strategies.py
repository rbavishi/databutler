import ast
import collections
import io
import textwrap
from contextlib import redirect_stdout
from typing import List, Dict, Optional, Collection, Union, Set, Tuple

import attrs

from databutler.mining.static_pandas_mining.autodoc_utils import (
    normalize_code_for_comparison,
    find_instantiation_map,
)
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import langmodels
from databutler.utils.logging import logger


@attrs.define(eq=False, repr=False)
class NLDescription:
    """Basic container for a natural language description of code"""

    #  The description that will be associated with the snippet in the database.
    primary_desc: str
    #  Additional NL that only helps the nl-to-code part of bidirectional consistency.
    auxiliary_descs: List[str]
    #  Additional (code) context that helps the nl-to-code part of bidirectional consistency.
    context: str

    @staticmethod
    def deserialize(v_dict: dict) -> "NLDescription":
        return NLDescription(
            primary_desc=v_dict["primary_desc"],
            auxiliary_descs=v_dict.get("auxiliary_descs", []),
            context=v_dict.get("context", ""),
        )

    def serialize(self) -> Dict:
        return {
            "primary_desc": self.primary_desc,
            "auxiliary_descs": self.auxiliary_descs,
            "context": self.context,
        }

    def pretty_print(self) -> str:
        auxiliary = "\n** ".join([""] + self.auxiliary_descs).strip()
        return (
            f"* {self.primary_desc}\n"
            f"{auxiliary}\n"
            f"{'Context: ' + self.context if self.context else ''}"
        )


@attrs.define(eq=False, repr=False)
class AutodocDescription:
    """Container for a single Autodoc description result after performing the bidirectional consistency check"""

    desc: NLDescription
    target_code: str
    target_template: str
    generated_code: str
    #  Was the generated code equivalent to the target code?
    equivalent: bool
    #  Was the parameterization successul?
    parameterized: bool
    #  How much assistance was provided for the second step of the bidirectional consistency check.
    #  0 = no assistance.
    assistance_level: int


@attrs.define(eq=False, repr=False)
class CanonicalAutodocDescription(AutodocDescription):
    parameterized_nl: Optional[str]
    parameterized_code: Optional[str]


@attrs.define(eq=False, repr=False)
class AutodocFewShotExample:
    """Basic container for a few-shot example"""

    code: str
    canonical: NLDescription
    nl_descs: List[NLDescription]
    parameterized_nl: str
    parameterized_code: str

    @staticmethod
    def deserialize(
        v_dict: Union[List[Dict], Dict]
    ) -> Union["AutodocFewShotExample", List["AutodocFewShotExample"]]:
        if isinstance(v_dict, list):
            return [AutodocFewShotExample.deserialize(x) for x in v_dict]

        return AutodocFewShotExample(
            code=v_dict["code"],
            canonical=NLDescription.deserialize(v_dict["canonical"]),
            nl_descs=[
                NLDescription.deserialize(nl_desc)
                for nl_desc in v_dict.get("nl_descs", [])
            ],
            parameterized_nl=v_dict["parameterized_nl"],
            parameterized_code=v_dict["parameterized_code"],
        )

    def serialize(self) -> Dict:
        return {
            "code": self.code,
            "canonical": self.canonical.serialize(),
            "nl_descs": [nl_desc.serialize() for nl_desc in self.nl_descs],
            "parameterized_nl": self.parameterized_nl,
            "parameterized_code": self.parameterized_code,
        }


def normalize_code_results(
    code_results: List[str],
    df_arg_names: Set[str],
    replace_singleton_lists: bool = True,
) -> List[Optional[str]]:
    normalized = []
    for code_result in code_results:
        try:
            #  Sometimes libcst messes up.
            ast.parse(code_result)
            code_result = normalize_code_for_comparison(
                code_result,
                df_arg_names,
                replace_singleton_lists=replace_singleton_lists,
            )
        except (astlib.cst.ParserSyntaxError, SyntaxError):
            normalized.append(None)
        else:
            normalized.append(code_result)

    return normalized


def check_parameterization(
    code: str,
    desc: str,
    parameterized_code: str,
    parameterized_nl: str,
) -> bool:
    """
    Check if the generated code is equivalent to the target code.
    """
    code_results = [code, parameterized_code]
    normalized_code_results = normalize_code_results(code_results, set())
    if normalized_code_results[0] is None or normalized_code_results[1] is None:
        return False

    code, parameterized_code = normalized_code_results
    param_code_ast = astlib.parse(parameterized_code)

    first_mod_stmt = next(iter(astlib.iter_body_stmts(param_code_ast)))
    #  It should be a function
    if not isinstance(first_mod_stmt, astlib.FunctionDef):
        return False

    #  With a single statement as the body
    if len(list(astlib.iter_body_stmts(first_mod_stmt))) != 1:
        return False

    first_func_stmt = next(iter(astlib.iter_body_stmts(first_mod_stmt)))
    #  It should be a return statement or a single expression statement
    if not isinstance(first_func_stmt, astlib.Return) and not isinstance(
        first_func_stmt, astlib.Expr
    ):
        return False
    else:
        param_expr = first_func_stmt.value

    params: List[str] = [param.name.value for param in first_mod_stmt.params.params]
    if not all(f"[{p}]" in parameterized_nl or p not in desc for p in params):
        #  Every parameter must be mentioned explicitly in the parameterized description
        return False

    #  There must be a valid instantiation for the original code
    inst_map = find_instantiation_map(param_expr, astlib.parse_expr(code))
    if not all(
        isinstance(k, astlib.Name) and k.value in params for k in inst_map.keys()
    ):
        return False

    return True


@attrs.define(eq=False, repr=False)
class CanonicalDescriptionsGenerator:
    """Container for logic to generate canonical descriptions"""

    task_description_desc_gen: str = (
        "Describe the following data science code snippets in plain english. "
        "Be as exhaustive as possible and repeat any variable or column references verbatim in double quotes. "
        "Do not quote any code fragment directly and explain every argument and lambda function if used."
    )
    task_description_code_gen: str = (
        "Generate a Python pandas code snippet given the english description. "
        "The snippet must be a python expression. All variables provided in the context should be used."
    )
    task_description_parameterization: str = (
        "Generalize the code and its natural language description into a reusable function that can be "
        "applied to other inputs. "
        "Clearly distinguish which arguments are column parameters. "
        "Variables provided as context need not be generalized. "
        "Ensure all the arguments are also mentioned in the parameterized natural language."
    )
    stop_token: str = "END"
    engine: str = "code-davinci-002"

    def create_desc_gen_prompt(
        self,
        target_snippet: MinedResult,
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool = True,
    ) -> str:
        """Create prompt to prime the model to generate descriptions given a code snippet.

        This is the first step of the bidirectional consistency check.
        """
        with redirect_stdout(io.StringIO()) as captured_stdout:
            #  Before the examples, describe the task. This does seem to improve the model's performance.
            print(self.task_description_desc_gen)
            print("")

            #  First, put in the few shot examples.
            for ex in few_shot_examples:
                print("Code:")
                print(ex.code.rstrip())
                print("")

                print("Description:")
                print("*", ex.canonical.primary_desc)
                if use_auxiliary_descs:
                    for desc in ex.canonical.auxiliary_descs:
                        print("*", desc)

                print("")
                print(self.stop_token)
                print("")

            #  Then, put in the target code.
            print("Code:")
            if (
                target_snippet.extra_context_vars is not None
                and len(target_snippet.extra_context_vars) > 0
            ):
                for k, v in target_snippet.extra_context_vars.items():
                    print(f"{k}: {v}")
            print(target_snippet.code.rstrip())
            print("")

            print("Description:")
            print("* ", end="")

            return captured_stdout.getvalue()

    def create_code_gen_prompt(
        self,
        target_desc: NLDescription,
        target_snippet: MinedResult,
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool = True,
    ) -> str:
        """Create prompt to prime the model to generate code given a code snippet.

        This is the second step of the bidirectional consistency check.
        """
        with redirect_stdout(io.StringIO()) as captured_stdout:
            #  Before the examples, describe the task. This does seem to improve the model's performance.
            print(self.task_description_code_gen)
            print("")

            #  First, put in the few shot examples.
            for ex in few_shot_examples:
                print("Description:")
                print("*", ex.canonical.primary_desc)
                if use_auxiliary_descs:
                    for desc in ex.canonical.auxiliary_descs:
                        print("*", desc)

                print("")
                print("Code:")
                print(ex.code.rstrip())
                print("")

                print(self.stop_token)
                print("")

            #  Then, put in the target description.
            print("Description:")
            print(f"* {target_desc.primary_desc}")
            if use_auxiliary_descs:
                for desc in target_desc.auxiliary_descs:
                    print(f"* {desc}")

            if (
                target_snippet.extra_context_vars is not None
                and len(target_snippet.extra_context_vars) > 0
            ):
                print("Context:")
                for k, v in target_snippet.extra_context_vars.items():
                    print(f"{k}: {v}")

                print("")

            print("Code:", end="")
            return captured_stdout.getvalue()

    def create_parameterization_prompt(
        self,
        target_desc: NLDescription,
        target_code: str,
        few_shot_examples: List[AutodocFewShotExample],
    ) -> str:
        """
        Create a prompt to prime the model to parameterize a previously generated description
        along with its code.
        """
        with redirect_stdout(io.StringIO()) as captured_stdout:
            #  Before the examples, describe the task. This does seem to improve the model's performance.
            print(self.task_description_parameterization)
            print("")

            #  First, put in the few shot examples.
            for ex in few_shot_examples:
                print("Description:")
                print(ex.canonical.primary_desc)
                print("")

                print("Code:")
                print(ex.code.rstrip())
                print("")

                print("Parameterized Description:")
                print(ex.parameterized_nl)
                print("")

                print("Parameterized Code:")
                print(ex.parameterized_code.rstrip())
                print("")

                print(self.stop_token)
                print("")

            #  Then, put in the target description.
            print("Description:")
            print(f"{target_desc.primary_desc.strip()}")
            print("")

            print("Code: ")
            print(target_code.rstrip())
            print("")

            print("Parameterized Description:", end="")
            return captured_stdout.getvalue()

    def generate_nl_candidates(
        self,
        snippets: List[MinedResult],
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool = True,
        max_tokens: Union[int, Collection[int]] = 96,
        num_candidates_per_target: int = 10,
        temperature: float = 0.5,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[List[NLDescription]]:
        #  Gather the prompts.
        prompts: List[str] = [
            self.create_desc_gen_prompt(snippet, few_shot_examples, use_auxiliary_descs)
            for snippet in snippets
        ]

        #  Batch it up into a single request.
        responses: List[
            langmodels.OpenAICompletionResponse
        ] = langmodels.openai_completion(
            engine=self.engine,
            prompts=prompts,
            temperature=temperature,
            num_completions=num_candidates_per_target,
            max_tokens=max_tokens if isinstance(max_tokens, int) else max(max_tokens),
            stop=[self.stop_token],
            key_manager=key_manager,
        )

        text_resps_list: List[List[str]] = [
            list({c.text.strip() for c in resp.completions}) for resp in responses
        ]

        #  Prepare NLDescription objects.
        nl_descs_list: List[List[NLDescription]] = []
        for snippet, text_resps in zip(snippets, text_resps_list):
            nl_descs: List[NLDescription] = []
            for text_resp in text_resps:
                #  The text should be in bullet point format. The first is the primary description. Rest are auxiliary.
                #  The first '*' is part of the prompt so no need to match against that.
                points = text_resp.split("\n* ")
                points = [point.strip() for point in points]
                #  Canonical descriptions do not have context.
                nl_descs.append(
                    NLDescription(
                        primary_desc=points[0], auxiliary_descs=points[1:], context=""
                    )
                )

            logger.opt(colors=True).debug(
                f"\n<blue>Generated Canonical Descriptions for Code:\n{snippet.code}</blue>\n"
                f"-------------------------------------\n"
            )
            for desc in nl_descs:
                logger.opt(colors=True, raw=True).debug(
                    f"<m>{desc.pretty_print()}</m>\n"
                )

            nl_descs_list.append(nl_descs)

        return nl_descs_list

    def verify_candidates(
        self,
        flattened_snippets: List[MinedResult],
        flattened_candidates: List[NLDescription],
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool,
        assistance_level: int,
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[CanonicalAutodocDescription]:
        """Perform the second step of the bidirectional consistency check."""
        assert len(flattened_candidates) == len(flattened_snippets)

        results: List[CanonicalAutodocDescription] = []
        if len(flattened_candidates) > batch_size:
            for idx in range(0, len(flattened_candidates), batch_size):
                results.extend(
                    self.verify_candidates(
                        flattened_snippets[idx : idx + batch_size],
                        flattened_candidates[idx : idx + batch_size],
                        few_shot_examples,
                        use_auxiliary_descs,
                        assistance_level,
                        batch_size=batch_size,
                        key_manager=key_manager,
                    )
                )

        else:
            #  Gather the prompts
            prompts: List[str] = [
                self.create_code_gen_prompt(
                    candidate,
                    snippet,
                    few_shot_examples,
                    use_auxiliary_descs=use_auxiliary_descs,
                )
                for snippet, candidate in zip(flattened_snippets, flattened_candidates)
            ]

            #  Compute the max tokens as the max of the target code lengths + 64
            max_tokens = max(
                [
                    len(langmodels.tokenize(snippet.code, self.engine)["token_ids"])
                    + 64
                    for snippet in flattened_snippets
                ]
            )

            #  Batch it up into a single request.
            responses: List[
                langmodels.OpenAICompletionResponse
            ] = langmodels.openai_completion(
                engine=self.engine,
                prompts=prompts,
                temperature=0.0,
                num_completions=1,
                max_tokens=max_tokens,
                stop=[self.stop_token],
                key_manager=key_manager,
            )

            generated_codes: List[str] = [
                resp.completions[0].text for resp in responses
            ]

            #  Perform normalization.
            df_vars: Set[str] = set()
            for snippet in flattened_snippets:
                df_vars.update(snippet.df_vars)

            normalized_codes: List[Optional[str]] = normalize_code_results(
                generated_codes, df_vars
            )
            target_codes: List[str] = [
                normalize_code_for_comparison(snippet.code, df_vars)
                for snippet in flattened_snippets
            ]

            for snippet, desc, generated_code, target_code in zip(
                flattened_snippets, flattened_candidates, normalized_codes, target_codes
            ):
                if generated_code is None:
                    equivalent = False
                else:
                    equivalent = generated_code == target_code

                logger.opt(colors=True, raw=True).debug("<e>Code Generation:</e>\n")
                logger.opt(colors=True, raw=True).debug(
                    f"<e>{desc.pretty_print()}</e>\n"
                )
                logger.opt(colors=True, raw=True).debug(
                    f"<e>Generated: {generated_code}\n</e>"
                )
                logger.opt(colors=True, raw=True).debug(
                    f"<e>Target: {target_code}\n</e>"
                )
                if equivalent:
                    logger.opt(colors=True, raw=True).debug(
                        f"<e>Equivalent: <g>{equivalent}</g></e>\n"
                    )
                else:
                    logger.opt(colors=True, raw=True).debug(
                        f"<e>Equivalent: <r>{equivalent}</r></e>\n"
                    )
                logger.opt(colors=True, raw=True).debug(
                    f"<e>Assistance Level: {assistance_level}</e>\n"
                )
                logger.opt(colors=True, raw=True).debug(
                    f"<e>--------------------------------------</e>\n"
                )

                results.append(
                    CanonicalAutodocDescription(
                        desc=desc,
                        target_code=target_code,
                        target_template=snippet.template,
                        generated_code=generated_code,
                        equivalent=equivalent,
                        #  Will update this later
                        parameterized=False,
                        assistance_level=assistance_level,
                        #  We will populate this later.
                        parameterized_nl=None,
                        parameterized_code=None,
                    )
                )

        return results

    def perform_generalization(
        self,
        descs: List[CanonicalAutodocDescription],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> None:
        if len(descs) > batch_size:
            for idx in range(0, len(descs), batch_size):
                self.perform_generalization(
                    descs[idx : idx + batch_size],
                    few_shot_examples,
                    batch_size=batch_size,
                    key_manager=key_manager,
                )

            return

        #  Gather the prompts
        prompts: List[str] = [
            self.create_parameterization_prompt(
                desc.desc, desc.target_code, few_shot_examples
            )
            for desc in descs
        ]

        #  Compute the max tokens as the max of the target code lengths + target nl lengths + 128
        max_tokens = max(
            [
                len(langmodels.tokenize(desc.target_code, self.engine)["token_ids"])
                + len(desc.desc.primary_desc)
                + 128
                for desc in descs
            ]
        )

        #  Batch it up into a single request.
        responses: List[
            langmodels.OpenAICompletionResponse
        ] = langmodels.openai_completion(
            engine=self.engine,
            prompts=prompts,
            temperature=0.0,
            num_completions=1,
            max_tokens=max_tokens,
            stop=[self.stop_token],
            key_manager=key_manager,
        )

        generated_texts: List[str] = [resp.completions[0].text for resp in responses]

        for desc, text in zip(descs, generated_texts):
            splitted = text.split("Parameterized Code:")
            if len(splitted) != 2:
                #  the model did not follow the format
                continue

            generated_param_nl, generated_param_code = splitted
            generated_param_nl = generated_param_nl.strip()
            generated_param_code = textwrap.dedent(generated_param_code).strip()
            logger.opt(colors=True, raw=True).debug(
                f"<y>Description: {desc.desc.primary_desc}</y>\n"
            )
            logger.opt(colors=True, raw=True).debug(
                f"<y>Code: {desc.target_code}\n</y>"
            )
            logger.opt(colors=True, raw=True).debug(
                f"<y>Parameterized NL: {generated_param_nl}</y>\n"
            )
            logger.opt(colors=True, raw=True).debug(
                f"<y>Parameterized Code: {generated_param_code}</y>\n"
            )

            #  Check if it is valid. That is, can it be instantiated in a way to obtain the target code back.
            #  Also check if all the variables are mentioned in the parameterized nl.
            is_valid = check_parameterization(
                desc.target_code,
                desc.desc.primary_desc,
                generated_param_code,
                generated_param_nl,
            )
            tag_open, tag_close = ("<g>", "</g>") if is_valid else ("<r>", "</r>")
            logger.opt(colors=True, raw=True).debug(
                f"<y>Validity: {tag_open}{is_valid}{tag_close}</y>\n"
            )

            if is_valid:
                desc.parameterized_nl = generated_param_nl
                desc.parameterized_code = generated_param_code
                desc.parameterized = True
            else:
                desc.parameterized = False

    def process_nl_candidates(
        self,
        snippets: List[MinedResult],
        candidates_list: List[List[NLDescription]],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[List[CanonicalAutodocDescription]]:
        to_process: List[
            Tuple[MinedResult, NLDescription, Optional[CanonicalAutodocDescription]]
        ] = []
        for snippet, candidates in zip(snippets, candidates_list):
            for candidate in candidates:
                to_process.append((snippet, candidate, None))

        autodoc_results: Dict[
            str, List[CanonicalAutodocDescription]
        ] = collections.defaultdict(list)
        parameterization_worklist: List[CanonicalAutodocDescription] = []

        for assistance_level in [0, 1]:
            if len(to_process) == 0:
                break

            if assistance_level == 0:
                use_auxiliary_descs = False
            elif assistance_level == 1:
                use_auxiliary_descs = True
            else:
                raise ValueError("Invalid assistance level")

            todo_snippets, todo_candidates, _ = list(zip(*to_process))

            results = self.verify_candidates(
                todo_snippets,
                todo_candidates,
                few_shot_examples,
                use_auxiliary_descs,
                assistance_level,
                batch_size=batch_size,
                key_manager=key_manager,
            )

            to_process.clear()
            for snippet, candidate, result in zip(
                todo_snippets, todo_candidates, results
            ):
                if not result.equivalent:
                    to_process.append((snippet, candidate, result))
                else:
                    autodoc_results[snippet.uid].append(result)
                    parameterization_worklist.append(result)

        #  Add the failure cases too.
        for snippet, candidate, result in to_process:
            if result is not None:
                autodoc_results[snippet.uid].append(result)

        #  Perform generalization
        if len(parameterization_worklist) > 0:
            self.perform_generalization(
                parameterization_worklist,
                few_shot_examples,
                batch_size=batch_size,
                key_manager=key_manager,
            )

        for snippet in snippets:
            logger.opt(raw=True).debug("Code Generation Results:\n")
            for result in autodoc_results[snippet.uid]:
                logger.opt(raw=True).debug(f"{result.desc.pretty_print()}\n")
                logger.opt(raw=True).debug(f"Generated: {result.generated_code}\n")
                logger.opt(raw=True).debug(f"Target: {result.target_code}\n")
                logger.opt(raw=True).debug(
                    f"Parameterized NL: {result.parameterized_nl}\n"
                )
                logger.opt(raw=True).debug(
                    f"Parameterized Code: {result.parameterized_code}\n"
                )

                if result.equivalent and result.parameterized:
                    logger.opt(colors=True, raw=True).debug(
                        f"Equivalence and Parameterization: <g>{True}</g>\n"
                    )
                else:
                    logger.opt(colors=True, raw=True).debug(
                        f"Equivalence and Parameterization: <r>{False}</r>\n"
                    )

                logger.opt(raw=True).debug(
                    f"Assistance Level: {result.assistance_level}\n"
                )
                logger.opt(raw=True).debug(f"--------------------------------------\n")

        return [autodoc_results[snippet.uid] for snippet in snippets]

    def generate(
        self,
        snippets: List[MinedResult],
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool = True,
        max_tokens: Union[int, Collection[int]] = 96,
        num_candidates_per_target: int = 10,
        temperature: float = 0.5,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[List[AutodocDescription]]:

        #  Generate the candidates
        candidates_list: List[List[NLDescription]] = self.generate_nl_candidates(
            snippets,
            few_shot_examples,
            use_auxiliary_descs,
            max_tokens,
            num_candidates_per_target,
            temperature,
            key_manager,
        )

        return self.process_nl_candidates(
            snippets,
            candidates_list,
            few_shot_examples,
            key_manager=key_manager,
        )
