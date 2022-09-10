import ast
import collections
import io
import textwrap
from contextlib import redirect_stdout
from typing import List, Dict, Optional, Collection, Union, Set, Tuple

import attrs

from databutler.mining.static_pandas_mining.autodoc_result import (
    AutodocResult,
    NLDescription,
    AutodocDescription,
    CanonicalAutodocDescription,
)
from databutler.mining.static_pandas_mining.autodoc_utils import (
    normalize_code_for_comparison,
    find_instantiation_map,
)
from databutler.mining.static_pandas_mining.mining_utils import MinedResult
from databutler.pat import astlib
from databutler.utils import langmodels
from databutler.utils.logging import logger


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
class DescriptionsGenerator:
    """Main container for autodoc logic"""

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

    def create_diverse_desc_gen_prompt(
        self,
        target_param_code: str,
        few_shot_examples: List[AutodocFewShotExample],
    ) -> str:
        """
        Create prompt to prime the model to generate diverse descriptions given a parameterized code snippet.
        """
        with redirect_stdout(io.StringIO()) as captured_stdout:
            #  Before the examples, describe the task. This does seem to improve the model's performance.
            print(self.task_description_desc_gen)
            print("")

            #  First, put in the few shot examples.
            for ex in few_shot_examples:
                print("Code:")
                print(ex.parameterized_code.rstrip())
                print("")

                print("Description:")
                for desc in ex.nl_descs:
                    print(f"* {desc.primary_desc}")

                print("")
                print(self.stop_token)
                print("")

            #  Then, put in the target code.
            print("Code:")
            print(target_param_code.rstrip())
            print("")

            print("Description:")
            print("* ", end="")

            return captured_stdout.getvalue()

    def create_diverse_code_gen_prompt(
        self,
        target_desc: NLDescription,
        few_shot_examples: List[AutodocFewShotExample],
        use_auxiliary_descs: bool = True,
    ) -> str:
        """
        Create prompt to prime the model to generate code given a diverse, less-precise description.
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
                print(ex.parameterized_code.rstrip())
                print("")

                print(self.stop_token)
                print("")

            #  Then, put in the target description.
            print("Description:")
            print(f"* {target_desc.primary_desc}")
            if use_auxiliary_descs:
                for desc in target_desc.auxiliary_descs:
                    print(f"* {desc}")

            print("Code:")
            print(target_desc.context, end="")

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
                    "<m>{desc}</m>\n", desc=desc.pretty_print()
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
                    "<e>{desc}</e>\n", desc=desc.pretty_print()
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

    def generate_diverse_description_candidates(
        self,
        worklist: List[Dict],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        num_candidates_per_item: int = 5,
        max_tokens: int = 32 * 3,
        temperature: float = 0.8,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[List[NLDescription]]:
        results: List[List[NLDescription]] = []
        if len(worklist) > batch_size:
            for idx in range(0, len(worklist), batch_size):
                results.extend(
                    self.generate_diverse_description_candidates(
                        worklist[idx : idx + batch_size],
                        few_shot_examples=few_shot_examples,
                        batch_size=batch_size,
                        num_candidates_per_item=num_candidates_per_item,
                        temperature=temperature,
                        key_manager=key_manager,
                    )
                )

            return results

        #  Gather the prompts
        prompts: List[str] = [
            self.create_diverse_desc_gen_prompt(item["param_code"], few_shot_examples)
            for item in worklist
        ]

        #  Batch it up into a single request.
        responses: List[
            langmodels.OpenAICompletionResponse
        ] = langmodels.openai_completion(
            engine=self.engine,
            prompts=prompts,
            temperature=temperature,
            num_completions=num_candidates_per_item,
            max_tokens=max_tokens,
            stop=[self.stop_token],
            key_manager=key_manager,
        )

        #  Everything should be in bullet point format.
        bullet_points_list: List[List[str]] = []
        for resp in responses:
            collected: Set[str] = set()
            for completion in resp.completions:
                collected.update(completion.text.strip().split("\n* "))

            bullet_points_list.append(list(collected))

        #  Prepare NLDescription objects.
        for item, bullet_points in zip(worklist, bullet_points_list):
            param_code = item["param_code"]

            #  First line (containing the signature) will be the context.
            context = param_code.strip().split("\n")[0]

            descs: List[NLDescription] = []
            for bullet_point in bullet_points:
                desc = NLDescription(
                    primary_desc=bullet_point,
                    auxiliary_descs=item["canonical_desc"].desc.auxiliary_descs,
                    context=context,
                )
                descs.append(desc)

            results.append(descs)

            logger.opt(raw=True).debug(
                f"Diverse description candidates for\n{param_code}\n"
            )
            for desc in descs:
                logger.opt(raw=True).debug(f"{desc.primary_desc}\n")

            logger.opt(raw=True).debug("-----------\n")

        return results

    def verify_diverse_desc_candidates(
        self,
        worklist: List[Dict],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> Dict[str, List[AutodocDescription]]:
        results: Dict[str, List[AutodocDescription]] = collections.defaultdict(list)
        if len(worklist) > batch_size:
            for idx in range(0, len(worklist), batch_size):
                for k, v in self.verify_diverse_desc_candidates(
                    worklist[idx : idx + batch_size],
                    few_shot_examples=few_shot_examples,
                    batch_size=batch_size,
                    key_manager=key_manager,
                ).items():
                    results[k].extend(v)

            return results

        for assistance_level in [0, 1]:
            if assistance_level == 0:
                use_auxiliary_descs = False
            else:
                use_auxiliary_descs = True

            #  Gather the prompts
            prompts: List[str] = [
                self.create_diverse_code_gen_prompt(
                    item["candidate"],
                    few_shot_examples,
                    use_auxiliary_descs=use_auxiliary_descs,
                )
                for item in worklist
            ]

            max_tokens = max(
                (
                    len(
                        langmodels.tokenize(item["param_code"], self.engine)[
                            "token_ids"
                        ]
                    )
                    for item in worklist
                )
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
            generated_texts: List[str] = [
                resp.completions[0].text.strip() for resp in responses
            ]

            #  Failures will go here for the next assistance level, if any.
            new_worklist: List[Dict] = []

            for item, generated_text in zip(worklist, generated_texts):
                param_code_body = item["param_code"].strip().split("\n")[1][4:]
                if param_code_body.startswith("return "):
                    param_code_body = param_code_body[7:].lstrip()

                if generated_text.startswith("return "):
                    generated_text = generated_text[7:].lstrip()

                normalized_code = normalize_code_results(
                    [generated_text], set(), replace_singleton_lists=False
                )[0]
                logger.opt(raw=True).debug(
                    f"Description: {item['candidate'].pretty_print()}\n"
                )
                open_tag = "<g>" if normalized_code == param_code_body else "<r>"
                close_tag = "</g>" if normalized_code == param_code_body else "</r>"
                logger.opt(raw=True, colors=True).debug(
                    f"Generated ({assistance_level}): <y>{normalized_code}</y>\n"
                    f"Target: <e>{param_code_body}</e>\n"
                    f"Success: {open_tag}{normalized_code == param_code_body}{close_tag}\n"
                    f"-------------------------------\n"
                )
                if normalized_code == param_code_body:
                    desc = AutodocDescription(
                        desc=item["candidate"],
                        assistance_level=assistance_level,
                        target_code=item["param_code"],
                        target_template="",
                        generated_code=generated_text,
                        equivalent=True,
                    )
                    results[item["snippet"].uid].append(desc)
                else:
                    new_worklist.append(item)

            worklist = new_worklist
            if len(worklist) == 0:
                break

        return results

    def generate_diverse_descriptions(
        self,
        snippets: List[MinedResult],
        canonical_descs: List[List[CanonicalAutodocDescription]],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> Dict[str, List[AutodocDescription]]:
        """
        Generate diverse descriptions from the canonical descriptions.
        """
        assert len(snippets) == len(canonical_descs)
        assert all(
            all(desc.parameterized for desc in descs) and len(descs) > 0
            for descs in canonical_descs
        )

        #  Prepare the worklist by getting all the unique parameterizations for each snippet.
        worklist: List[Dict] = []
        for snippet, descs in zip(snippets, canonical_descs):
            #  Uniqify using the parameterized code.
            uniqified_descs: List[CanonicalAutodocDescription] = list(
                {
                    desc.parameterized_code.replace("    return ", "    "): desc
                    for desc in descs
                }.values()
            )
            for desc in uniqified_descs:
                worklist.append(
                    {
                        "snippet": snippet,
                        "param_code": desc.parameterized_code,
                        "canonical_desc": desc,
                    }
                )

        #  Generate the diverse description candidates
        desc_candidates: List[
            List[NLDescription]
        ] = self.generate_diverse_description_candidates(
            worklist,
            few_shot_examples=few_shot_examples,
            batch_size=batch_size,
            key_manager=key_manager,
        )

        #  Perform the bidirectional consistency check for these candidates.
        verification_worklist: List[Dict] = [
            {
                "snippet": item["snippet"],
                "param_code": item["param_code"],
                "canonical_desc": item["canonical_desc"],
                "candidate": cand,
            }
            for cands, item in zip(desc_candidates, worklist)
            for cand in cands
        ]

        verified_descs: Dict[
            str, List[AutodocDescription]
        ] = self.verify_diverse_desc_candidates(
            verification_worklist,
            few_shot_examples=few_shot_examples,
            batch_size=batch_size,
            key_manager=key_manager,
        )

        return verified_descs

    def process_nl_candidates(
        self,
        snippets: List[MinedResult],
        candidates_list: List[List[NLDescription]],
        few_shot_examples: List[AutodocFewShotExample],
        batch_size: int = 10,
        key_manager: Optional[langmodels.OpenAIKeyManager] = None,
    ) -> List[AutodocResult]:
        to_process: List[
            Tuple[MinedResult, NLDescription, Optional[CanonicalAutodocDescription]]
        ] = []
        for snippet, candidates in zip(snippets, candidates_list):
            for candidate in candidates:
                to_process.append((snippet, candidate, None))

        canonical_descs: Dict[
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
                    canonical_descs[snippet.uid].append(result)
                    parameterization_worklist.append(result)

        #  Add the failure cases too.
        for snippet, candidate, result in to_process:
            if result is not None:
                canonical_descs[snippet.uid].append(result)

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
            for result in canonical_descs[snippet.uid]:
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

        autodoc_results: Dict[str, AutodocResult] = {}
        for snippet in snippets:
            success_descs: List[AutodocDescription] = []
            failed_descs: List[AutodocDescription] = []
            for desc in canonical_descs[snippet.uid]:
                (
                    success_descs
                    if desc.equivalent and desc.parameterized
                    else failed_descs
                ).append(desc)

            autodoc_results[snippet.uid] = AutodocResult(
                uid=snippet.uid,
                code=snippet.code,
                template=snippet.template,
                success=len(success_descs) > 0,
                canonical_descs=success_descs,
                #  We will populate this later
                additional_descs=[],
                failed_descs=failed_descs,
            )

        #  Prepare worklist for diverse description generation.
        diverse_desc_worklist: List[
            Tuple[MinedResult, List[CanonicalAutodocDescription]]
        ] = []
        for snippet in snippets:
            descs = canonical_descs[snippet.uid]
            successful_descs = [
                desc for desc in descs if desc.parameterized and desc.equivalent
            ]
            if len(successful_descs) > 0:
                diverse_desc_worklist.append((snippet, successful_descs))

        #  Generate diverse descriptions.
        if len(diverse_desc_worklist) > 0:
            todo_snippets, todo_descs_list = list(zip(*diverse_desc_worklist))
            diverse_descs = self.generate_diverse_descriptions(
                todo_snippets,
                todo_descs_list,
                few_shot_examples,
                batch_size=batch_size,
                key_manager=key_manager,
            )
            for uid, descs in diverse_descs.items():
                autodoc_results[uid].additional_descs = descs

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
    ) -> List[AutodocResult]:

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


@attrs.define(eq=False, repr=False)
class DiverseDescriptionGenerator:
    task_description_desc_gen: str = (
        "Create diverse but complete descriptions of the code snippets below. "
        "Do not repeat the style or manner of descriptions across bullet points. "
        "Cover all the operations in the description but no need to reference or explain variables and parameters."
    )
    task_description_code_gen: str = ""

    stop_token: str = "END"
    engine: str = "code-davinci-002"

    def create_desc_gen_prompt(
        self,
        target_param_code: str,
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
                for desc in ex.nl_descs:
                    print(f"* {desc.primary_desc}")

                print("")
                print(self.stop_token)
                print("")

            #  Then, put in the target code.
            print("Code:")
            print(target_param_code.rstrip())
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
