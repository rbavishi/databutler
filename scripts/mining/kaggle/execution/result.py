from enum import Enum
from typing import Optional, Dict

import attrs


class NotebookExecStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2


class NotebookExecErrorType(Enum):
    NO_ERROR = 0
    CONTAINER_START_ERROR = 1
    RUNTIME_EXCEPTION = 2


@attrs.define(eq=False)
class NotebookExecResult:
    status: NotebookExecStatus
    error_type: NotebookExecErrorType
    msg: str
    metadata: Optional[Dict] = None
