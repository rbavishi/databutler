from typing import Union, List

import attrs


@attrs.define
class FewShotExampleCodeAndNL:
    """
    A few-shot example that serves code-to-NL and NL-to-code tasks.
    """
    code: str
    nl: Union[str, List[str]]


@attrs.define
class FewShotExampleCodeChangeAndNL:
    """
    A few-shot example that serves code-change-to-NL and NL-to-code-change tasks.
    """
    old_code: str
    new_code: str
    nl: Union[str, List[str]]
