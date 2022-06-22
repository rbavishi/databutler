import ast
import inspect
from typing import List


def parse(code, wrap_module=False):
    if wrap_module:
        result = ast.parse(code)
    else:
        result = ast.parse(code).body[0]

    return result


def parse_obj(obj):
    src = inspect.getsource(obj).strip()
    return parse(src)


def parse_file(fname: str) -> ast.Module:
    with open(fname, "r") as f:
        return ast.parse(f.read().strip())


def get_all_names(n: ast.AST) -> List[str]:
    res = []
    for node in ast.walk(n):
        if isinstance(node, ast.Name):
            res.append(node.id)

    return res
