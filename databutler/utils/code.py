import ast
from typing import Any

import astunparse
import black

from databutler.pat import astlib


class _CodeOptimizer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name):
            if node.targets[0].id == node.value.id:
                return None

        return node


def normalize_code(code: str) -> str:
    """
    Returns a formatting-normalized version of the code by running through a parser and then a code-generator.

    Args:
        code: A string corresponding to the code to normalize

    Returns:
        (str): The normalized code.
    """
    return unparse_native_ast(ast.parse(code))


def unparse_native_ast(code_ast: ast.AST) -> str:
    """
    Returns a formatting-normalized version of the provided Python AST by running through a code-generator.

    Args:
        code_ast: The Python AST to unparse.

    Returns:
        (str): A normalized code string.
    """
    mode = black.FileMode()
    return black.format_str(astunparse.unparse(code_ast).strip(), mode=mode).strip()


def unparse_astlib_ast(code_ast: astlib.AstNode) -> str:
    """
    Returns a formatting-normalized version of the provided AstLib AstNode by running through a code-generator.

    Args:
        code_ast: The AST obtained via astlib to unparse.

    Returns:
        (str): A normalized code string.
    """
    return normalize_code(astlib.to_code(code_ast))


def optimize_code(code: str) -> str:
    """
    Performs simple optimizations such as removing obvious no-ops such as statements which assign a variable to itself.

    Args:
        code: A string corresponding to the code to optimize.

    Returns:
        (str): The optimized code.
    """
    code_ast = ast.parse(code)
    mode = black.FileMode()
    return black.format_str(astunparse.unparse(_CodeOptimizer().visit(code_ast)), mode=mode).strip()
