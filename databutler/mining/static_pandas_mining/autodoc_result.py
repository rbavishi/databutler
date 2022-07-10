from typing import List, Dict, Optional

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
    #  How much assistance was provided for the second step of the bidirectional consistency check.
    #  0 = no assistance.
    assistance_level: int


@attrs.define(eq=False, repr=False)
class CanonicalAutodocDescription(AutodocDescription):
    #  Was the parameterization successul?
    parameterized: bool
    parameterized_nl: Optional[str]
    parameterized_code: Optional[str]


@attrs.define(eq=False, repr=False)
class AutodocResult:
    uid: str
    code: str
    template: str
    success: bool
    canonical_descs: List[CanonicalAutodocDescription]
    additional_descs: List[AutodocDescription]
    failed_descs: List[AutodocDescription]
    is_derived: bool = False
