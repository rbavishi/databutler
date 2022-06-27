from typing import List, Dict

import attrs


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
    def from_json(json_dict: dict) -> "NLDescription":
        return NLDescription(
            primary_desc=json_dict["primary_desc"],
            auxiliary_descs=json_dict.get("auxiliary_descs", []),
            context=json_dict.get("context", ""),
        )

    def to_json(self) -> Dict:
        return {
            "primary_desc": self.primary_desc,
            "auxiliary_descs": self.auxiliary_descs,
            "context": self.context,
        }


@attrs.define(eq=False, repr=False)
class AutodocFewShotExample:
    """Basic container for a few-shot example"""

    code: str
    canonicals: List[NLDescription]
    nl_descs: List[NLDescription]

    @staticmethod
    def from_json(json_dict: dict) -> "AutodocFewShotExample":
        return AutodocFewShotExample(
            code=json_dict["code"],
            canonicals=[
                NLDescription.from_json(nl_desc) for nl_desc in json_dict["canonical"]
            ],
            nl_descs=[
                NLDescription.from_json(nl_desc) for nl_desc in json_dict["nl_descs"]
            ],
        )

    def to_json(self) -> Dict:
        return {
            "code": self.code,
            "canonical": [nl_desc.to_json() for nl_desc in self.canonicals],
            "nl_descs": [nl_desc.to_json() for nl_desc in self.nl_descs],
        }


@attrs.define(eq=False, repr=False)
class CanonicalDescriptionsGenerator:
    """Container for logic to generate canonical descriptions"""

    task_description: str = ("Describe the following data science code snippets in plain english. "
    "Be as exhaustive as possible and repeat any constants verbatim in double quotes. ")

    def create_prompt(
        self, target_code: str, few_shot_examples: List[AutodocFewShotExample]
    ) -> str:

        raise NotImplementedError()

    def generate(
        self, target_code_strs: List[str], few_shot_examples: List[AutodocFewShotExample]
    ) -> List[List[NLDescription]]:
        raise NotImplementedError()
