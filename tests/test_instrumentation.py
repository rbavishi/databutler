import collections
import itertools
import unittest
from typing import Dict, List

import attr
import numpy as np

from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import ExprWrappersGenerator, ExprWrapper, Instrumentation, Instrumenter, \
    StmtCallback, \
    StmtCallbacksGenerator, ExprCallbacksGenerator, ExprCallback, CallDecoratorsGenerator, CallDecorator

_DUMMY_GLOBAL = 10


class InstrumentationTests(unittest.TestCase):
    def test_fstrings_1(self):
        class Dummy(ExprWrappersGenerator):
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._dummy, name=self.gen_wrapper_id()))

                return wrappers

            def _dummy(self, value):
                return value

        instrumentation = Instrumentation(expr_wrapper_gens=[Dummy()])
        instrumenter = Instrumenter(instrumentation=instrumentation)

        def func():
            col = 12
            print(f'{col:11}  missing values')

        _, func_1 = instrumenter.process_func(func)
        func_1()

    def test_concatenated_string_1(self):
        discovered = set()

        class Dummy(ExprWrappersGenerator):
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._dummy, name=self.gen_wrapper_id()))

                return wrappers

            def _dummy(self, value):
                discovered.add(value)
                return value

        instrumentation = Instrumentation(expr_wrapper_gens=[Dummy()])
        instrumenter = Instrumenter(instrumentation=instrumentation)

        def func():
            col = 12
            print(f"{col}" "defg")

        _, func_1 = instrumenter.process_func(func)
        func_1()
        self.assertIn(12, discovered)
        self.assertIn("12defg", discovered)
        self.assertNotIn("defg", discovered)

    def test_wrappers_1(self):
        cnt = 0

        @attr.s(cmp=False, repr=False)
        class Dummy(ExprWrappersGenerator):
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._dummy, name=self.gen_wrapper_id(),
                                                   arg_str='globals(), locals()'))

                return wrappers

            def _dummy(self, value, d_globals, d_locals):
                nonlocal cnt
                cnt += d_locals['a']
                return value

        def func(a):
            return a

        instrumentation = Instrumentation.from_generators(Dummy())
        instrumenter = Instrumenter(instrumentation=instrumentation)

        _, func1 = instrumenter.process_func(func)
        func1(10)
        self.assertEqual(10, cnt)

    def test_wrappers_2(self):
        cnt = 0

        @attr.s(cmp=False, repr=False)
        class Dummy(ExprWrappersGenerator):
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._dummy2, name=self.gen_wrapper_id(),
                                                   arg_str='globals(), locals()'))
                    wrappers[n].append(ExprWrapper(callable=self._dummy2, name=self.gen_wrapper_id(),
                                                   arg_str='globals(), locals()'))
                    wrappers[n].append(ExprWrapper(callable=self._dummy1, name=self.gen_wrapper_id(),
                                                   arg_str=''))

                return wrappers

            def _dummy1(self, value):
                return value

            def _dummy2(self, value, d_globals, d_locals):
                nonlocal cnt
                cnt += d_locals['a']
                return value

        def func(a):
            return a

        instrumentation = Instrumentation.from_generators(Dummy())
        instrumenter = Instrumenter(instrumentation=instrumentation)

        _, func1 = instrumenter.process_func(func)
        func1(10)
        self.assertEqual(20, cnt)

    def test_decorators_1(self):
        @attr.s(cmp=False, repr=False)
        class Dummy(ExprWrappersGenerator, ExprCallbacksGenerator):
            def gen_expr_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprCallback]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprCallback(callable=self._dummy_callback, name=self.gen_expr_callback_id(),
                                                    arg_str='', position='post'))

                return wrappers

            def _dummy_callback(self):
                pass

            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._dummy, name=self.gen_wrapper_id()))

                return wrappers

            def _dummy(self, value):
                return value

        def func():
            import functools
            decorator = functools.lru_cache()

            @decorator
            def f(a):
                return a * 2

            print(f(2), f(2))

        instrumentation = Instrumentation.from_generators(Dummy())
        instrumenter = Instrumenter(instrumentation=instrumentation)

        a, func = instrumenter.process_func(func)
        func()

    def test_expr_intercept_1(self):

        @attr.s(cmp=False, repr=False)
        class NumpyInt64Converter(ExprWrappersGenerator):
            """
            Intercepts all integer values, and replaces them with a np.int64 equivalent.
            """
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for expr in self.iter_valid_exprs(ast_root):
                    wrappers[expr].append(ExprWrapper(callable=self._boxing, name=self.gen_wrapper_id()))

                return wrappers

            def _boxing(self, value):
                if isinstance(value, int) and (not isinstance(value, bool)):
                    return np.int64(value)

                return value

        mydict = {}

        def func():
            global _DUMMY_GLOBAL
            _DUMMY_GLOBAL = 20
            mydict['a'] = _DUMMY_GLOBAL + 30

        instrumentation = Instrumentation(expr_wrapper_gens=[NumpyInt64Converter()])
        instrumenter = Instrumenter(instrumentation=instrumentation)
        new_ast, func = instrumenter.process_func(func)
        func()
        self.assertIsInstance(mydict['a'], np.int64)

    def test_mandatory_post_callbacks_1(self):
        counter: int = 0

        class NonMandatory(StmtCallbacksGenerator):
            def gen_stmt_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
                callbacks: Dict[astlib.AstStatementT, List[StmtCallback]] = {}
                for n in self.iter_stmts(ast_root):
                    callbacks[n] = [StmtCallback(callable=self._callback, name=self.gen_stmt_callback_id(),
                                                 position='post', mandatory=False)]

                return callbacks

            def _callback(self, d_globals, d_locals):
                nonlocal counter
                counter += 1

        class Mandatory(StmtCallbacksGenerator):
            def gen_stmt_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.AstStatementT, List[StmtCallback]]:
                callbacks: Dict[astlib.AstStatementT, List[StmtCallback]] = {}
                for n in self.iter_stmts(ast_root):
                    callbacks[n] = [StmtCallback(callable=self._callback, name=self.gen_stmt_callback_id(),
                                                 position='post', mandatory=True)]

                return callbacks

            def _callback(self, d_globals, d_locals):
                nonlocal counter
                counter += 1

        instrumentation_1 = Instrumentation(stmt_callback_gens=[NonMandatory()])
        instrumentation_2 = Instrumentation(stmt_callback_gens=[Mandatory()])
        instrumenter_1 = Instrumenter(instrumentation=instrumentation_1)
        instrumenter_2 = Instrumenter(instrumentation=instrumentation_2)

        def func():
            c = 0
            for i in [1, 2]:
                c += i
                if i > 0:
                    break
            print(c)

        func_1 = instrumenter_1.process_func(func)[1]
        func_2 = instrumenter_2.process_func(func)[1]
        counter = 0
        func_1()
        ctr_1 = counter
        counter = 0
        func_2()
        ctr_2 = counter

        self.assertEqual(4, ctr_1)
        self.assertEqual(6, ctr_2)  # Should be counting the if-statement and the break statement.

    def test_expr_callbacks_1(self):
        trace = []

        @attr.s(cmp=False, repr=False)
        class MyTraversal(ExprCallbacksGenerator):
            _memory: Dict[astlib.BaseExpression, int] = attr.ib(init=False, factory=dict)

            def gen_expr_callbacks(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprCallback]]:
                callbacks: Dict[astlib.BaseExpression, List[ExprCallback]] = collections.defaultdict(list)
                for expr in self.iter_valid_exprs(ast_root):
                    for k, v in itertools.chain(self.gen_pre_callback(expr).items(),
                                                self.gen_post_callback(expr).items()):
                        callbacks[k].extend(v)

                return callbacks

            def gen_pre_callback(self, expr):
                def cb(d_globals, d_locals):
                    if expr not in self._memory:
                        self._memory[expr] = len(self._memory)

                    trace.append(self._memory[expr])

                return {expr: [ExprCallback(callable=cb, name=self.gen_expr_callback_id(),
                                            position='pre')]}

            def gen_post_callback(self, expr):
                def cb(d_globals, d_locals):
                    trace.append(self._memory[expr])

                return {expr: [ExprCallback(callable=cb, name=self.gen_expr_callback_id(),
                                            position='post')]}

        instrumentation = Instrumentation(expr_callback_gens=[MyTraversal()])
        instrumenter = Instrumenter(instrumentation=instrumentation)

        def func(a, b):
            return (a + 1) + b

        new_ast, instrumented_func = instrumenter.process_func(func)
        res = instrumented_func(10, 20)
        self.assertEqual(31, res)
        self.assertEqual([0, 1, 2, 2, 3, 3, 1, 4, 4, 0], trace)

    def test_process_func_1(self):
        num_exprs_counter: int = 0

        class NumExprCounter(ExprWrappersGenerator):
            def gen_expr_wrappers(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[ExprWrapper]]:
                wrappers: Dict[astlib.BaseExpression, List[ExprWrapper]] = collections.defaultdict(list)

                for n in self.iter_valid_exprs(ast_root):
                    wrappers[n].append(ExprWrapper(callable=self._expr_counter, name=self.gen_wrapper_id()))

                return wrappers

            def _expr_counter(self, value):
                nonlocal num_exprs_counter
                num_exprs_counter += 1
                return value

        instrumentation = Instrumentation(expr_wrapper_gens=[NumExprCounter()])
        instrumenter = Instrumenter(instrumentation=instrumentation)

        def func_target(a, b):
            c = a + b
            return c

        def func_target_with_imports(a, b):
            import numpy as npy
            from numpy import nan
            c = npy.int64(a) + npy.int64(b)
            d = nan + nan
            return c

        def func_target_with_closure(a, b):
            c = np.int64(a) + np.int64(b)
            return c

        new_ast, instrumented_func = instrumenter.process_func(func_target)
        new_ast, instrumented_func_with_imports = instrumenter.process_func(func_target_with_imports)
        new_ast, instrumented_func_with_closure = instrumenter.process_func(func_target_with_closure)

        self.assertEqual(30, instrumented_func(10, 20))
        self.assertGreater(num_exprs_counter, 0)

        num_exprs_counter = 0

        self.assertEqual(30, instrumented_func_with_imports(10, 20))
        self.assertGreater(num_exprs_counter, 0)

        num_exprs_counter = 0

        self.assertEqual(30, instrumented_func_with_closure(10, 20))
        self.assertGreater(num_exprs_counter, 0)

    def test_interception_1(self):
        @attr.s(cmp=False, repr=False)
        class CallInterceptor(CallDecoratorsGenerator):
            """
            A very simple interceptor that intercepts all calls to functions named 'b'
            and returns 0 as the result if the first argument ('intercept') is True.
            """

            def gen_decorators(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
                decorators: Dict[astlib.BaseExpression, List[CallDecorator]] = collections.defaultdict(list)

                for call in self.iter_calls(ast_root):
                    #  Note that the key is call.func, not call itself.
                    decorators[call.func].append(self._gen_decorator(call))

                return decorators

            def _gen_decorator(self, call: astlib.Call) -> CallDecorator:
                def wrapper(func, args, kwargs):
                    if func.__name__ == "b" and args[0] is True:
                        return 0

                    return func(*args, **kwargs)

                return CallDecorator(callable=wrapper, does_not_return=False)

        mydict = {}

        def func():
            def a(arg1, arg2):
                return arg1 + arg2

            def b(intercept, arg2):
                return arg2 * 2

            mydict['first'] = a(1, 2)
            mydict['second'] = a('1', '2')
            mydict['third'] = b(True, 3)  # Should be intercepted
            mydict['fourth'] = b(False, '2')  # Should not be intercepted

        instrumentation = Instrumentation(call_decorator_gens=[CallInterceptor()])
        instrumenter = Instrumenter(instrumentation=instrumentation)
        _, func = instrumenter.process_func(func)
        func()
        self.assertEqual(3, mydict['first'])
        self.assertEqual('12', mydict['second'])
        self.assertEqual(0, mydict['third'])
        self.assertEqual('22', mydict['fourth'])
