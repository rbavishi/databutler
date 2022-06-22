import builtins
import inspect
from typing import Any, List, Callable, Tuple

from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder_utils import (
    TraceItemsCollector,
    TraceEventsCollector,
)
from databutler.pat.analysis.hierarchical_trace import specs
from databutler.pat.analysis.hierarchical_trace.specs import SPECS


def _get_func_search_key(func: Callable):
    if inspect.ismethod(func) or hasattr(func, "__self__"):
        self_ = getattr(func, "__self__", None)
        if self_ is not None and self_ is not builtins:
            #  This is a bound method, return the underlying function
            if hasattr(func, "__func__"):
                return func.__func__
            if hasattr(self_.__class__, func.__name__):
                return getattr(self_.__class__, func.__name__)

    elif (not inspect.isroutine(func)) and (not inspect.isclass(func)):
        #  Probably an object with a __call__ method
        if hasattr(func, "__call__") and hasattr(func, "__class__"):
            func = getattr(func.__class__, "__call__")

    return func


def ignore_func(func, ret_val, args, kwargs):
    """
    Whether to ignore checking of specs for this function.
    :param func:
    :param ret_val:
    :param args:
    :param kwargs:
    :return:
    """
    key = _get_func_search_key(func)

    if key not in SPECS:
        #  This means that we are being unsound. Any function for which we don't have the spec,
        #  we consider it non side-effecting.
        #  TODO : Is there an easy way to whitelist a large number of things, and switch to being conservative,
        #  TODO : i.e. mark a function as side-effecting on all arguments if it isn't found in the spec.
        return True

    return False


def has_spec(func, ret_val, args, kwargs) -> bool:
    """
    Whether a spec exists for the passed function
    :param func:
    :param ret_val:
    :param args:
    :param kwargs:
    :return:
    """
    key = _get_func_search_key(func)
    return key in SPECS


def check_spec(func, ret_val, args, kwargs) -> List[Tuple[Any, str]]:
    if inspect.ismethod(func) or hasattr(func, "__self__"):
        self_ = getattr(func, "__self__", None)
        if self_ is not None and self_ is not builtins:
            #  This is a bound method, return the underlying function
            if hasattr(func, "__func__"):
                func = func.__func__
                args = [self_] + list(args)
            elif hasattr(self_.__class__, func.__name__):
                func = getattr(self_.__class__, func.__name__)
                args = [self_] + list(args)

    elif (not inspect.isroutine(func)) and (not inspect.isclass(func)):
        #  Probably an object with a __call__ method
        if hasattr(func, "__call__") and hasattr(func, "__class__"):
            self_ = func
            func = getattr(self_.__class__, "__call__")
            args = [self_] + list(args)

    try:
        sig = inspect.signature(func)
        binding = sig.bind(*args, **kwargs)
        binding.apply_defaults()

    except ValueError:
        #  Okay looks like this might be an issue with Python 3.6 and below for builtin methods.
        #  For now mark everything as written.
        #  TODO : Improve precision
        return [*((i, "write") for i in args), *((v, "write") for v in kwargs.values())]

    specs = SPECS[func]
    reads_and_writes = []
    for spec in specs:
        reads_and_writes.extend(spec(binding, ret_val))

    return reads_and_writes


def setup(
    clock: LogicalClock,
    trace_events_collector: TraceEventsCollector,
    trace_items_collector: TraceItemsCollector,
):
    specs._CLOCK = clock
    specs._TRACE_EVENTS_COLLECTOR = trace_events_collector
    specs._TRACE_ITEMS_COLLECTOR = trace_items_collector
