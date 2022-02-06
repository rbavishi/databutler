import ast
from typing import Set, Optional, Collection

from databutler.pat import astlib


class _NodeRemovalTransformer(ast.NodeTransformer):
    def __init__(self, to_remove: Collection[ast.AST]):
        self._to_remove: Set[ast.AST] = set(to_remove)

    def visit(self, node: ast.AST) -> Optional[ast.AST]:
        if node in self._to_remove:
            return None
        else:
            return self.generic_visit(node)


def remove_nodes_from_native_ast(code_ast: ast.AST, to_remove: Collection[ast.AST]) -> ast.AST:
    return _NodeRemovalTransformer(to_remove).visit(code_ast)


def remove_nodes_from_astlib_ast(code_ast: astlib.AstNode, to_remove: Collection[astlib.AstNode]) -> astlib.AstNode:
    return astlib.remove_nodes_from_ast(code_ast, to_remove)
