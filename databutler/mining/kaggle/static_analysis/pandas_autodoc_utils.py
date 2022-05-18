import collections
from typing import Deque, Tuple

from databutler.pat import astlib


def find_instantiation_map(template_ast: astlib.AstNode, code_ast: astlib.AstNode):
    worklist: Deque[Tuple[astlib.AstNode, astlib.AstNode]] = collections.deque()
    worklist.append((template_ast, code_ast))

    req_mapping = {}

    while len(worklist) > 0:
        t_node, c_node = worklist.popleft()

        if not isinstance(t_node, type(c_node)):
            req_mapping[t_node] = c_node
            continue

        t_children = t_node.children
        c_children = c_node.children

        if len(t_children) != len(c_children):
            req_mapping[t_node] = c_node
            continue

        if len(t_children) == 0 and not t_node.deep_equals(c_node):
            # if astlib.to_code(t_node) != "":
            req_mapping[t_node] = c_node
            continue

        for t_c, c_c in zip(t_children, c_children):
            worklist.append((t_c, c_c))

    for k, v in req_mapping.items():
        print(f"{astlib.to_code(k)} maps to {astlib.to_code(v)}")
