import io
from typing import List, Dict, Optional, Any, Set

import attrs
import libcst
import matplotlib.pyplot as plt

from databutler.datana.viz.utils import mpl_exec
from databutler.pat import astlib
from databutler.utils.logging import logger


def _is_syntactically_correct(code: str) -> bool:
    try:
        astlib.parse(code)
        return True
    except:
        return False


def _has_undefined_references(code: str) -> bool:
    wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
    scopes = set(wrapper.resolve(libcst.metadata.ScopeProvider).values())

    for scope in scopes:
        for access in scope.accesses:
            if len(access.referents) == 0:
                return True

    return False


def _check_statically(code: str) -> bool:
    if not _is_syntactically_correct(code):
        return False

    if _has_undefined_references(code):
        return False

    return True


def _get_viz_as_bytes(code: str, args: List[Any], kw_args: Dict[str, Any],
                      timeout: Optional[int] = None) -> Optional[bytes]:
    try:
        fig = mpl_exec.run_viz_code_matplotlib_mp(code, pos_args=args, kw_args=kw_args, func_name='viz',
                                                  timeout=timeout)
        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            plt.close(fig)
            return buf.read()

        return None

    except:
        return None


@attrs.define(eq=False, repr=False)
class Minimizer:
    code_ast: astlib.FunctionDef
    args: List[Any]
    kw_args: Dict[str, Any]
    eligible_stmts: List[astlib.AstStatementT]
    orig_fig: bytes
    timeout_per_run: Optional[int] = 5

    _seen: Dict[str, bool] = attrs.field(init=False, factory=dict)
    _num_executions: int = attrs.field(init=False, default=0)

    def _create_candidate(self, to_keep: List[astlib.AstStatementT]) -> str:
        new_candidate = astlib.update_stmt_body(self.code_ast, astlib.prepare_body(to_keep))
        new_code = astlib.to_code(new_candidate)

        return new_code

    def _check_candidate(self, code: str) -> bool:
        if code in self._seen:
            return self._seen[code]

        if not _check_statically(code):
            return False

        self._num_executions += 1
        res = self._seen[code] = _get_viz_as_bytes(code, self.args, self.kw_args,
                                                   timeout=self.timeout_per_run) == self.orig_fig
        return res

    def _run_delta_debugging(self) -> str:
        n = 2
        self._seen.clear()
        self._num_executions = 0

        logger.info("Starting Minimization")
        eligible_stmts = self.eligible_stmts[:]

        while True:
            num_per_bin = (len(eligible_stmts) + n - 1) // n

            bins = []
            for i in range(0, len(eligible_stmts), num_per_bin):
                bins.append(eligible_stmts[i: i + num_per_bin])

            for b in bins:
                new_body = b[:]
                new_code = self._create_candidate(new_body)

                if self._check_candidate(new_code):
                    kept_stmts = "\n".join(astlib.to_code(s) for s in b)
                    logger.info(f"Retaining statements:\n{kept_stmts}")
                    eligible_stmts = new_body[:]
                    n = 2
                    break

            else:
                #  Did not break out of the loop, so try the complements.
                for b in bins:
                    new_body = [s for s in eligible_stmts if s not in b]
                    new_code = self._create_candidate(new_body)

                    if self._check_candidate(new_code):
                        removed_stmts = "\n".join(astlib.to_code(s) for s in b)
                        logger.info(f"Removing statements:\n{removed_stmts}")
                        n = max(n - 1, 2)
                        eligible_stmts = new_body[:]
                        break

                else:
                    #  Did not break out of the loop, so try to increase granularity if you can.
                    if n < len(eligible_stmts):
                        n = n * 2
                    else:
                        #  Can't do more, return what we have.
                        break

        logger.info(f"Finished minimization (Num-Execs: {self._num_executions})")
        return self._create_candidate(eligible_stmts)

    def _run_single_stmt_removal(self) -> str:
        removed: Set[astlib.AstStatementT] = set()

        logger.info("Starting Minimization")
        #  Go in reverse to adhere to the topological dependency order.
        for stmt in reversed(self.eligible_stmts):
            new_body = [s for s in self.eligible_stmts if s is not stmt and s not in removed]
            new_code = self._create_candidate(new_body)

            if self._check_candidate(new_code):
                logger.info(f"Removing statement:\n{astlib.to_code(stmt)}")
                removed.add(stmt)

        logger.info(f"Finished minimization (Num-Execs: {self._num_executions})")
        new_body = [s for s in self.eligible_stmts if s not in removed]
        return self._create_candidate(new_body)

    def run(self) -> str:
        if len(self.eligible_stmts) <= 25:
            #  For small-enough code, removing one at a time can be better than delta-debugging.
            return self._run_single_stmt_removal()
        else:
            return self._run_delta_debugging()


def minimize_code(code: str, args: List[Any], kw_args: Dict[str, Any], timeout_per_run: Optional[int] = 5) -> str:
    orig_fig = _get_viz_as_bytes(code, args, kw_args, timeout=timeout_per_run)
    if orig_fig is None:
        return code

    ast_root = astlib.parse(code)
    assert isinstance(ast_root, astlib.Module)
    func_def = next(iter(astlib.iter_body_stmts(ast_root)))
    assert isinstance(func_def, astlib.FunctionDef)

    minimizer = Minimizer(func_def,
                          args=args, kw_args=kw_args,
                          eligible_stmts=list(astlib.iter_body_stmts(func_def)),
                          orig_fig=orig_fig,
                          timeout_per_run=timeout_per_run)

    return minimizer.run()
