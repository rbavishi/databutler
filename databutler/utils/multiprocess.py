import multiprocessing as mp
from typing import Callable, Optional

from pebble import concurrent


class FuncTimeoutError(TimeoutError):
    pass


def run_func_in_process(func: Callable, *args, _timeout: Optional[int] = None, **kwargs):
    c_func = concurrent.process(timeout=_timeout, context=mp.get_context('spawn'))(func)
    future = c_func(*args, **kwargs)

    try:
        result = future.result()
        return result

    except TimeoutError:
        raise FuncTimeoutError
