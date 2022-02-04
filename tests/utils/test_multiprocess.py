import unittest

from databutler.utils import multiprocess


def _square_fn(n: int) -> int:
    return n * n


class MultiProcessTests(unittest.TestCase):
    def test_separate_process_1(self):
        self.assertEqual(100, multiprocess.run_func_in_process(_square_fn, 10, _timeout=10))

    def test_parallel_tasks_1(self):
        tasks = [1, 2, 3, 4]
        task_results = multiprocess.run_tasks_in_parallel(_square_fn, tasks)
        self.assertTrue(all(tr.is_success() for tr in task_results))
        self.assertEqual([1, 4, 9, 16], [tr.result for tr in task_results])
