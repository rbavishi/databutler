import collections
from typing import Set, Dict, List

import attr

from databutler.pat import astlib
from databutler.pat.analysis.instrumentation import CallDecoratorsGenerator, \
    CallDecorator


@attr.s(cmp=False, repr=False)
class IPythonMagicBlocker(CallDecoratorsGenerator):
    to_block: Set[str] = attr.ib(factory=set)

    def gen_decorators(self, ast_root: astlib.AstNode) -> Dict[astlib.BaseExpression, List[CallDecorator]]:
        decorators: Dict[astlib.BaseExpression, List[CallDecorator]] = collections.defaultdict(list)

        for call_expr in self.iter_calls(ast_root):
            if "run_line_magic" in astlib.to_code(call_expr.func):
                decorators[call_expr.func].append(CallDecorator(callable=self._intercept_magic,
                                                                does_not_return=False))

        return decorators

    def _intercept_magic(self, func, args, kwargs):
        if args[0] in self.to_block:
            return

        return func(*args, **kwargs)
