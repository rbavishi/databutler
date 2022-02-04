import multiprocessing as mp
import traceback
from enum import Enum
from typing import Callable, Optional, Dict, Any, List

import attrs
from pebble import concurrent, ProcessPool, ProcessExpired

from databutler.utils.logging import logger


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


class TaskRunStatus(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2
    PROCESS_EXPIRED = 3


@attrs.define(eq=False, repr=False)
class TaskResult:
    status: TaskRunStatus

    result: Optional[Any] = None
    exception_tb: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == TaskRunStatus.SUCCESS

    def is_timeout(self) -> bool:
        return self.status == TaskRunStatus.TIMEOUT

    def is_exception(self) -> bool:
        return self.status == TaskRunStatus.EXCEPTION

    def is_process_expired(self) -> bool:
        return self.status == TaskRunStatus.PROCESS_EXPIRED


def run_tasks_in_parallel(func: Callable,
                          tasks: List[Any],
                          timeout_per_task: Optional[int] = None,
                          max_tasks_per_worker: Optional[int] = None,
                          num_workers: int = 2,
                          use_spawn: bool = True) -> List[TaskResult]:
    """

    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        timeout_per_task: The timeout, in seconds, to use per task.
        max_tasks_per_worker: Maximum number of tasks assigned to a single process / worker. None means infinite.
            Use 1 to force a restart.
        num_workers: Maximum number of parallel workers.
        use_spawn: The 'spawn' multiprocess context is used if True. 'fork' is used otherwise.

    Returns:
        A list of TaskResult objects, one per task.
    """

    mode = 'spawn' if use_spawn else 'fork'
    task_results: List[TaskResult] = []

    with ProcessPool(max_workers=num_workers,
                     max_tasks=0 if max_tasks_per_worker is None else max_tasks_per_worker,
                     context=mp.get_context(mode)) as pool:
        future = pool.map(func, tasks, timeout=timeout_per_task)

        iterator = future.result()

        while True:
            try:
                result = next(iterator)

            except StopIteration:
                break

            except TimeoutError as error:
                logger.warning(f"Process timed out after {error.args[1]} seconds")
                task_results.append(TaskResult(
                    status=TaskRunStatus.TIMEOUT,
                ))
            except ProcessExpired as error:
                logger.warning(f"Process exited with code {error.exitcode}: {str(error)}")
                task_results.append(TaskResult(
                    status=TaskRunStatus.PROCESS_EXPIRED,
                ))
            except Exception as error:
                logger.exception(error)
                exception_tb = traceback.format_exc()

                task_results.append(TaskResult(
                    status=TaskRunStatus.EXCEPTION,
                    exception_tb=exception_tb,
                ))

            else:
                task_results.append(TaskResult(
                    status=TaskRunStatus.SUCCESS,
                    result=result,
                ))

        return task_results
