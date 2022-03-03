from enum import Enum
from typing import Optional, Dict

import attrs


class KaggleExecStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2


@attrs.define(eq=False)
class KaggleExecResult:
    status: KaggleExecStatus
    msg: str
    metadata: Optional[Dict] = None
