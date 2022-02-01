import multiprocessing as mp
from typing import Callable, Optional

from pebble import concurrent


class FuncTimeoutError(TimeoutError):
    pass


def run_func_in_process(func: Callable, *args, _timeout: Optional[int] = None, _use_spawn: bool = True, **kwargs):
    """
    Runs the provided function in a separate process with the supplied args and kwargs. The args, kwargs, and
    return values must all be pickle-able.

    Args:
        func: The function to run.
        *args: Positional args, if any.
        _timeout: A timeout to use for the function.
        _use_spawn: The 'spawn' multiprocess context is used if True. 'fork' is used otherwise.
        **kwargs: Keyword args, if any.

    Returns:
        The result of executing the function.
    """
    mode = 'spawn' if _use_spawn else 'fork'
    c_func = concurrent.process(timeout=_timeout, context=mp.get_context(mode))(func)
    future = c_func(*args, **kwargs)

    try:
        result = future.result()
        return result

    except TimeoutError:
        raise FuncTimeoutError
