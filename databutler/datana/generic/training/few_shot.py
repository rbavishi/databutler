from typing import Union, List

import attrs


@attrs.define
class FewShotExampleCodeAndNL:
    """
    A few-shot example that serves code-to-NL and NL-to-code tasks.
    """
    code: str
    nl: Union[str, List[str]]
