"""
Core elements of the hierarchical program trace.
"""
import collections
from abc import ABC
from enum import Enum, auto
from typing import Optional, List, Dict, Hashable, Set, Iterator

import attr
import intervaltree
from sortedcontainers import SortedKeyList

from databutler.pat import astlib
from databutler.utils import caching


class TraceItemType(Enum):
    STMT = auto()
    EXPR = auto()
    AUX = auto()


class DependencyType(Enum):
    READ_AFTER_WRITE = auto()
    DEF_ACCESS = auto()


@attr.s(cmp=False, repr=False, slots=True)
class TraceItem:
    """
    The basic unit of the hierarchical program trace.
    """

    start_time: int = attr.ib()
    end_time: int = attr.ib()

    #  Node corresponding to the trace item.
    ast_node: astlib.AstNode = attr.ib()
    #  Scope in which this trace item was generated.
    scope_id: astlib.ScopeId = attr.ib()

    #  Type of the item. Auxiliary items are generated to capture semantics
    #  of compound statements and function calls. See `builder_aux` for examples.
    item_type: TraceItemType = attr.ib()
    # Obj-ID of the result of the expression if the item is of type EXPR.
    obj_id: int = attr.ib(default=None)

    #  The immediate parent item.
    par_item: Optional["TraceItem"] = attr.ib(default=None)

    #  Miscellaneous meta-data. Especially useful for auxiliary statements.
    metadata: Dict = attr.ib(default=None)

    def is_stmt_type(self):
        """
        Convenience function for checking if the item is of statement type.
        :return:
        """
        return self.item_type == TraceItemType.STMT

    def is_expr_type(self):
        """
        Convenience function for checking if the item is of expression type.
        :return:
        """
        return self.item_type == TraceItemType.EXPR

    def is_auxiliary_type(self):
        """
        Convenience function for checking if the item is auxiliary.
        :return:
        """
        return self.item_type == TraceItemType.AUX


@attr.s(cmp=False, repr=False, slots=True)
class TraceEvent(ABC):
    """
    Base class for a PDG event.
    """

    timestamp: int = attr.ib()
    owner: Optional[TraceItem] = attr.ib()
    ast_node: astlib.AstNode = attr.ib()
    obj_id: int = attr.ib()


@attr.s(cmp=False, repr=False, slots=True)
class ObjReadEvent(TraceEvent):
    """
    Generated whenever an object is encountered.
    We care about being able to reproduce *entire* objects, so we treat every encounter
    as a read of *all* the attributes, then their attributes, and so on recursively.
    Hence we do not track the precise attributes of the objects that are actually accessed.
    """


@attr.s(cmp=False, repr=False, slots=True)
class ObjWriteEvent(TraceEvent):
    """
    Generated whenever a write to an object (to any attribute) is detected.
    In line with the read-event, we do not track what is precisely modified as we
    care about reproducing object state in its entirety.
    """


@attr.s(cmp=False, repr=False, slots=True)
class DefEvent(TraceEvent):
    """
    Generated when a variable is assigned to (local variables, function defs, class defs, param definitions).
    Python does not differentiate between declaration and definition, so we do the same.
    """

    name: str = attr.ib()
    scope_id: astlib.ScopeId = (
        attr.ib()
    )  # ID of the scope in which the definition was done.
    def_key: Hashable = (
        attr.ib()
    )  # A unique object representing the variable this definition is for.


@attr.s(cmp=False, repr=False, slots=True)
class AccessEvent(TraceEvent):
    """
    Generated when a variable is accessed (local/global variables, functions, classes, params etc.).
    """

    name: str = attr.ib()
    scope_id: astlib.ScopeId = (
        attr.ib()
    )  # ID of the scope in which the variable was accessed.
    def_event: Optional[
        DefEvent
    ] = attr.ib()  # The event corresponding to the definition for this variable.


@attr.s(cmp=False, repr=False, slots=True)
class Dependency:
    src: TraceEvent = attr.ib()
    dst: TraceEvent = attr.ib()
    type: DependencyType = attr.ib()


@attr.s(cmp=False, repr=False)
class HierarchicalTrace:
    events: List[TraceEvent] = attr.ib()
    items: List[TraceItem] = attr.ib()

    # ----------------------- #
    #  Query Methods
    # ----------------------- #

    def get_end_time(self) -> int:
        """

        :return:
        """
        return self._default_item_end_time

    def get_events(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> List[TraceEvent]:
        """

        :param start_time:
        :param end_time:
        :return:
        """
        if start_time is None and end_time is None:
            return self.events[:]

        start_time = start_time or 0
        end_time = end_time or self._default_event_end_time
        return list(i.data for i in self._tree_events.envelop(start_time, end_time))

    def get_items(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> List[TraceItem]:
        """

        :param start_time:
        :param end_time:
        :return:
        """
        if start_time is None and end_time is None:
            return self.items[:]

        start_time = start_time or 0
        end_time = end_time or self._default_item_end_time
        return list(i.data for i in self._tree_items.envelop(start_time, end_time))

    def get_enveloping_items(
        self, start_time: int, end_time: int = None
    ) -> List[TraceItem]:
        """

        :param start_time:
        :param end_time:
        :return:
        """
        if end_time is None:
            return [i.data for i in self._tree_items.at(start_time)]

        return [
            i.data
            for i in self._tree_items.at(start_time) & self._tree_items.at(end_time)
        ]

    def get_parent(self, trace_item: TraceItem) -> Optional[TraceItem]:
        """

        :param trace_item:
        :return:
        """
        return trace_item.par_item

    def iter_parents(self, item: TraceItem) -> Iterator[TraceItem]:
        """

        :param item:
        :return:
        """
        cur = item.par_item
        while cur is not None:
            yield cur
            cur = cur.par_item

    def iter_children(self, item: TraceItem) -> Iterator[TraceItem]:
        """

        :param item:
        :return:
        """
        for i in self._direct_children_map[item]:
            yield i
            yield from self.iter_children(i)

    def get_direct_children(self, item: TraceItem) -> List[TraceItem]:
        """

        :param item:
        :return:
        """
        return self._direct_children_map[item]

    @caching.caching_method
    def get_internal_dependencies(self, item: TraceItem) -> List[Dependency]:
        """

        :param item:
        :return:
        """
        return [i.data for i in self._deps_tree.envelop(item.start_time, item.end_time)]

    @caching.caching_method
    def get_external_dependencies(self, item: TraceItem) -> List[Dependency]:
        """

        :param item:
        :return:
        """
        left_index = self._deps_dst_timestamps.bisect_key_left(item.start_time)
        right_index = self._deps_dst_timestamps.bisect_key_left(item.end_time)
        return [
            d
            for d in self._deps_dst_timestamps[left_index:right_index]
            if d.src.timestamp < item.start_time
        ]

    @caching.caching_method
    def get_dependencies(self, item: TraceItem) -> List[Dependency]:
        """

        :param item:
        :return:
        """
        return self.get_external_dependencies(item) + self.get_internal_dependencies(
            item
        )

    @caching.caching_method
    def get_explicitly_resolving_items(self, dependency: Dependency) -> List[TraceItem]:
        """

        :param dependency:
        :return:
        """
        if dependency.type == DependencyType.READ_AFTER_WRITE:
            return [
                i
                for i in self.get_enveloping_items(dependency.src.timestamp)
                if i.end_time <= dependency.dst.timestamp
            ]

        elif dependency.type == DependencyType.DEF_ACCESS:
            assert isinstance(dependency.src, DefEvent)
            return [
                i
                for i in self._items_exposing_defs[dependency.src]
                if i.end_time <= dependency.dst.timestamp
            ]

        else:
            raise ValueError(f"Unrecognized dependency of type {dependency.type}")

    @caching.caching_method
    def get_implicitly_resolving_items(self, dependency: Dependency) -> List[TraceItem]:
        """

        :param dependency:
        :return:
        """
        return list(
            self.get_enveloping_items(
                dependency.src.timestamp, dependency.dst.timestamp
            )
        )

    @caching.caching_method
    def get_resolving_items(self, dependency: Dependency) -> List[TraceItem]:
        """

        :param dependency:
        :return:
        """
        return self.get_explicitly_resolving_items(
            dependency
        ) + self.get_implicitly_resolving_items(dependency)

    @caching.caching_method
    def get_affording_items(self, event: TraceEvent) -> List[TraceItem]:
        """

        :param event:
        :return:
        """

        if isinstance(event, (ObjReadEvent, ObjWriteEvent, AccessEvent)):
            return [i for i in self.get_enveloping_items(event.timestamp)]

        elif isinstance(event, DefEvent):
            return self._items_exposing_defs[event]

    def get_def_events(self, obj_id: int, start_time: int = None, end_time: int = None):
        """

        :param obj_id:
        :param start_time:
        :param end_time:
        :return:
        """
        if start_time is None:
            start_time = -1
        if end_time is None:
            end_time = self.get_end_time()

        return [
            d
            for d in self._def_events_by_obj_id[obj_id]
            if start_time <= d.timestamp < end_time
        ]

    def get_expr_items(
        self,
        obj_id: int,
        min_start_time: int = None,
        max_start_time: int = None,
        min_end_time: int = None,
        max_end_time: int = None,
    ):
        """

        :param obj_id:
        :param min_start_time:
        :param max_start_time:
        :param min_end_time:
        :param max_end_time:
        :return:
        """
        if min_start_time is None:
            min_start_time = -1
        if min_end_time is None:
            min_end_time = -1
        if max_start_time is None:
            max_start_time = self.get_end_time()
        if max_end_time is None:
            max_end_time = self.get_end_time()

        return [
            d
            for d in self._expr_items_by_obj_id[obj_id]
            if min_start_time <= d.start_time < max_start_time
            and min_end_time <= d.end_time < max_end_time
        ]

    @caching.caching_method
    def get_objs_item_depends_on(self, item: TraceItem) -> Set[int]:
        """

        :param item:
        :return:
        """
        res: Set[int] = set()
        res.update(
            e.obj_id
            for e in self.get_events(start_time=item.start_time, end_time=item.end_time)
            if isinstance(e, ObjReadEvent)
        )

        for dep in self.get_external_dependencies(item):
            if dep.src.owner is not None:
                resolving_item = dep.src.owner
            else:
                resolving_items = sorted(
                    self.get_resolving_items(dep), key=lambda x: x.end_time
                )
                if len(resolving_items) > 0:
                    resolving_item = resolving_items[0]
                else:
                    resolving_item = None

            if resolving_item is not None:
                res.update(self.get_objs_item_depends_on(resolving_item))

        return res

    # --------------------------------------- #
    #  Data-structures for answering queries
    # --------------------------------------- #

    @caching.cached_property
    def _default_event_end_time(self) -> int:
        return max((e.timestamp for e in self.events), default=-1) + 1

    @caching.cached_property
    def _default_item_end_time(self) -> int:
        return max(i.end_time for i in self.items)

    @caching.cached_property
    def _tree_events(self) -> intervaltree.IntervalTree:
        return intervaltree.IntervalTree(
            intervaltree.Interval(begin=e.timestamp, end=e.timestamp + 1, data=e)
            for e in self.events
        )

    @caching.cached_property
    def _tree_items(self) -> intervaltree.IntervalTree:
        return intervaltree.IntervalTree(
            intervaltree.Interval(begin=i.start_time, end=i.end_time, data=i)
            for i in self.items
        )

    @caching.cached_property
    def _direct_children_map(self) -> Dict[TraceItem, List[TraceItem]]:
        res: Dict[TraceItem, List[TraceItem]] = collections.defaultdict(list)

        for item in self.items:
            if item.par_item is not None:
                res[item.par_item].append(item)

        return res

    @caching.cached_property
    def _dependencies(self) -> List[Dependency]:
        dependencies: List[Dependency] = []
        last_write: Dict[int, ObjWriteEvent] = {}
        for e in self.events:
            if isinstance(e, ObjWriteEvent):
                last_write[e.obj_id] = e

            elif isinstance(e, ObjReadEvent):
                if e.obj_id in last_write:
                    dependencies.append(
                        Dependency(
                            src=last_write[e.obj_id],
                            dst=e,
                            type=DependencyType.READ_AFTER_WRITE,
                        )
                    )

            elif isinstance(e, AccessEvent):
                if e.def_event is not None:
                    dependencies.append(
                        Dependency(
                            src=e.def_event, dst=e, type=DependencyType.DEF_ACCESS
                        )
                    )

        return dependencies

    @caching.cached_property
    def _deps_tree(self) -> intervaltree.IntervalTree:
        interval_class = intervaltree.Interval
        return intervaltree.IntervalTree(
            [
                interval_class(d.src.timestamp, d.dst.timestamp + 1, data=d)
                for d in self._dependencies
            ]
        )

    @caching.cached_property
    def _deps_src_timestamps(self) -> SortedKeyList:
        return SortedKeyList(self._dependencies, key=lambda x: x.src.timestamp)

    @caching.cached_property
    def _deps_dst_timestamps(self) -> SortedKeyList:
        return SortedKeyList(self._dependencies, key=lambda x: x.dst.timestamp)

    @caching.cached_property
    def _items_exposing_defs(self) -> Dict[DefEvent, List[TraceItem]]:
        last_def_map: Dict[
            TraceItem, Dict[Hashable, DefEvent]
        ] = collections.defaultdict(dict)
        items_exposing_defs: Dict[DefEvent, List[TraceItem]] = collections.defaultdict(
            list
        )
        for e in self.events:
            if isinstance(e, DefEvent):
                if e.owner is not None:
                    for i in self.get_enveloping_items(e.timestamp):
                        if i.scope_id == e.scope_id:
                            last_def_map[i][e.name, e.scope_id] = e

        for i, v in last_def_map.items():
            for e in v.values():
                items_exposing_defs[e].append(i)

        return items_exposing_defs

    @caching.cached_property
    def _def_events_by_obj_id(self) -> Dict[int, List[DefEvent]]:
        def_events_by_obj_id: Dict[int, List[DefEvent]] = collections.defaultdict(list)
        for e in self.events:
            if isinstance(e, DefEvent):
                def_events_by_obj_id[e.obj_id].append(e)

        return def_events_by_obj_id

    @caching.cached_property
    def _expr_items_by_obj_id(self) -> Dict[int, List[TraceItem]]:
        expr_items_by_obj_id: Dict[int, List[TraceItem]] = collections.defaultdict(list)
        for i in self.items:
            if i.is_expr_type():
                expr_items_by_obj_id[i.obj_id].append(i)

        return expr_items_by_obj_id
