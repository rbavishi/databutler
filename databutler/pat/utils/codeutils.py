import ast
import json
import textwrap
import inspect
import builtins
from typing import Any, Union, Dict, List
from inspect import ismodule, iscode, ClosureVars
from typing import Optional

from databutler.pat.utils import astutilities
from nbconvert.filters import ipython2python


class __automl_wrapped_ipython:
    """
    Wrap special IPython calls with exception handling.
    """

    def __init__(self):
        from IPython import get_ipython

        self.ipython = get_ipython()

    def __getattr__(self, item):
        if self.ipython is None:

            def func(*args, **kwargs):
                return

            return func

        val = getattr(self.ipython, item)
        if inspect.ismethod(val):

            def wrapper(*args, **kwargs):
                try:
                    return val(*args, **kwargs)
                except:
                    return None

            return wrapper

        return val


def convert_python_notebook_magics(src: Union[str, Dict[str, Any]]) -> Dict:
    if isinstance(src, str):
        src: Dict[str, Any] = json.loads(src)
    else:
        src = src.copy()

    if "cells" not in src:
        raise AssertionError("Did not recognize notebook format")

    for c in src["cells"]:
        if c["cell_type"] == "code":
            c_code = c["source"]
            if isinstance(c_code, list):
                c_code = "".join(c_code)

            c_code = ipython2python(c_code)
            c_code = c_code.replace("get_ipython", "__automl_wrapped_ipython")
            c["source"] = [i + "\n" for i in c_code.split("\n")]

    src["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "outputs": [],
            "metadata": {},
            "source": [
                "import IPython, inspect\n",
                "IPython.display.display_html = lambda *objs, **kwargs : None\n",
                *inspect.getsourcelines(__automl_wrapped_ipython)[0],
            ],
        },
    )

    return src


def convert_python_notebook_to_code(src: Union[str, Dict[str, Any]]) -> str:
    if isinstance(src, str):
        src: Dict[str, Any] = json.loads(src)
    else:
        src = src.copy()

    code: List[str] = []
    for c in src["cells"]:
        if c["cell_type"] == "code":
            c_code = c["source"]
            code.extend(c_code)

    return "".join(code)


def getclosurevars_recursive(func, f_ast: Optional[ast.FunctionDef] = None):
    """
    The default getclosurevars doesn't go over nested function defs and list comprehensions.
    We write a recursive version of the same.
    The logic is borrowed from this post - https://bugs.python.org/issue34947
    Args:
        f_ast (Optional[ast.FunctionDef]): The AST of the function if available.
            If not, an attempt will be made to retrieve the AST
        func (Callable): The function to inspect
    Returns:
        An instance of ClosureVars
    """
    f_code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    if func.__closure__ is None:
        nonlocal_vars = {}
    else:
        nonlocal_vars = {
            var: cell.cell_contents
            for var, cell in zip(f_code.co_freevars, func.__closure__)
        }

    annotation_names = []
    try:
        if f_ast is None:
            f_ast: ast.FunctionDef = astutilities.parse(
                textwrap.dedent(inspect.getsource(func))
            )

        for n in ast.walk(f_ast.args):
            if isinstance(n, ast.arg) and n.annotation is not None:
                annotation_names.extend(astutilities.get_all_names(n.annotation))
        if f_ast.returns is not None:
            annotation_names.extend(astutilities.get_all_names(f_ast.returns))

    except:
        pass

    # Global and builtin references are named in co_names and resolved
    # by looking them up in __globals__ or __builtins__
    global_ns = func.__globals__
    builtin_ns = global_ns.get("__builtins__", builtins.__dict__)
    if ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__

    global_vars = {}
    builtin_vars = {}
    unbound_names = set()
    codes = [f_code]
    while codes:
        #  The logic is recursive but is implemented iteratively
        code = codes.pop()
        for name in code.co_names:
            if name in ("None", "True", "False"):
                # Because these used to be builtins instead of keywords, they
                # may still show up as name references. We ignore them.
                continue
            try:
                global_vars[name] = global_ns[name]
            except KeyError:
                try:
                    builtin_vars[name] = builtin_ns[name]
                except KeyError:
                    unbound_names.add(name)

        for const in code.co_consts:
            #  Add the code to inspect recursively
            if iscode(const):
                codes.append(const)

    for name in annotation_names:
        try:
            global_vars[name] = global_ns[name]
        except KeyError:
            try:
                builtin_vars[name] = builtin_ns[name]
            except KeyError:
                unbound_names.add(name)

    return ClosureVars(nonlocal_vars, global_vars, builtin_vars, unbound_names)
