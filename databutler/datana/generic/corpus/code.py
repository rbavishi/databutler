import copy
from typing import Optional, List, Any, Dict

import attrs


@attrs.define(eq=False, repr=False)
class DatanaFunction:
    """
    The basic unit of a corpus - a code snippet corresponding to a function.

    Attributes:
        code_str (str): The raw code string corresponding to the function. Must include the function signature.
        uid (str): A string corresponding to a unique identifier for the function.
        func_name (str): A string representing the name of the function.
        pos_args (Optional[List[Any]]): An optional list of positional arguments associated with the
            function by default.
        kw_args (Optional[Dict[str, Any]]: An optional dictionary of keyword arguments associated with the function
            by default.
        metadata (Optional[Dict[str, Any]]: An optional dictionary corresponding to extra metadata about the function.
    """
    code_str: str
    uid: str

    func_name: str
    pos_args: Optional[List[Any]] = None
    kw_args: Optional[Dict[str, Any]] = None

    metadata: Optional[Dict] = None

    def copy(self):
        return DatanaFunction(
            code_str=self.code_str,
            uid=self.uid,
            func_name=self.func_name,
            # Only make shallow copies of the arguments
            pos_args=list(self.pos_args) if self.pos_args is not None else None,
            # Only make shallow copies of the arguments
            kw_args=self.kw_args.copy() if self.kw_args is not None else None,
            metadata=copy.deepcopy(self.metadata),  # Deepcopy metadata so it's safe for modification
        )
