from abc import ABC, abstractmethod
from typing import Any, Collection, List, Tuple, Set, Dict, Hashable, Union

import attrs


@attrs.define(eq=False, repr=False)
class ObjRef(ABC):
    """
    A wrapper around an object that can be used to retrieve the object on demand.
    """

    @abstractmethod
    def resolve(self) -> Any:
        """
        Retrieves the object from a resource.

        Must be defined by all implementing classes.
        """


@attrs.define(eq=False, repr=False)
class LazyCollection(ABC):
    """
    Collections over ObjRefs that make it easy to retrieve lists, tuples, sets, and dicts of obj-refs.
    """

    @abstractmethod
    def resolve(self) -> Collection:
        """
        Returns the collection built by retrieving the constituent objects from their refs.

        Implemented separately for lists, tuples, sets and dicts.
        """


@attrs.define(eq=False, repr=False)
class LazyList(LazyCollection):
    """
    A lazy list.
    """

    refs: List[ObjRef]

    def resolve(self) -> List:
        return [r.resolve() for r in self.refs]


@attrs.define(eq=False, repr=False)
class LazyTuple(LazyCollection):
    """
    A lazy tuple.
    """

    refs: Tuple[ObjRef]

    def resolve(self) -> Tuple:
        return tuple(r.resolve() for r in self.refs)


@attrs.define(eq=False, repr=False)
class LazySet(LazyCollection):
    """
    A lazy set.
    """

    refs: Set[ObjRef]

    def resolve(self) -> Set:
        return {r.resolve() for r in self.refs}


@attrs.define(eq=False, repr=False)
class LazyDict(LazyCollection):
    """
    A lazy dictionary. Only the values can be lazy, not the keys.
    """

    refs: Dict[Hashable, Union[Any, ObjRef]]

    def resolve(self) -> Dict:
        return {
            k: (v.resolve() if isinstance(v, ObjRef) else v)
            for k, v in self.refs.items()
        }
