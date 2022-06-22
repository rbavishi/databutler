import copy
from typing import Optional, List, Any, Dict, Union

import attrs

from databutler.utils import lazyobjs


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
    pos_args: Optional[Union[lazyobjs.LazyList, lazyobjs.ObjRef, List[Any]]] = None
    kw_args: Optional[Union[lazyobjs.LazyDict, lazyobjs.ObjRef, Dict[str, Any]]] = None

    metadata: Optional[Dict] = None

    def copy(self):
        if isinstance(self.pos_args, (lazyobjs.LazyList, lazyobjs.ObjRef)):
            new_pos_args = self.pos_args
        else:
            # Only make shallow copies of the arguments
            if self.pos_args is None:
                new_pos_args = self.pos_args
            else:
                # Only make shallow copies of the arguments
                new_pos_args = list(self.pos_args)

        if isinstance(self.kw_args, (lazyobjs.LazyDict, lazyobjs.ObjRef)):
            new_kw_args = self.kw_args
        else:
            if self.kw_args is None:
                new_kw_args = self.kw_args
            else:
                # Only make shallow copies of the arguments
                new_kw_args = self.kw_args.copy()

        return DatanaFunction(
            code_str=self.code_str,
            uid=self.uid,
            func_name=self.func_name,
            pos_args=new_pos_args,
            kw_args=new_kw_args,
            metadata=copy.deepcopy(
                self.metadata
            ),  # Deepcopy metadata so it's safe for modification
        )

    def get_pos_args(self) -> List[Any]:
        if isinstance(self.pos_args, (lazyobjs.LazyList, lazyobjs.ObjRef)):
            return self.pos_args.resolve()
        else:
            return self.pos_args

    def get_kw_args(self) -> Dict[str, Any]:
        if isinstance(self.kw_args, (lazyobjs.LazyDict, lazyobjs.ObjRef)):
            return self.kw_args.resolve()
        else:
            return self.kw_args
