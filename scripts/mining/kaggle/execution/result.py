from enum import Enum
from typing import Optional, Dict

import attrs


class NotebookExecStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2


@attrs.define(eq=False)
class NotebookExecResult:
    status: NotebookExecStatus
    msg: str
    metadata: Optional[Dict] = None
