import ast
import collections
from abc import ABC, abstractmethod
from typing import List, Union, Dict

import attrs

from databutler.datana.generic.corpus.code_changes.utils import astutils
from databutler.pat import astlib
from databutler.utils import code as codeutils


@attrs.define(eq=False, repr=False)
class BaseCodeChange(ABC):
    @classmethod
    @abstractmethod
    def apply_changes(cls, code: str, changes: List['BaseCodeChange']) -> str:
        """
        Class method to apply the provided list of changes to the supplied code.

        Why supply all changes at once? This makes it easy to reason about all possible effects together.
        For AST-based changes, this also helps us ensure all the changes can potentially access the same AST at once.

        Args:
            code: A string corresponding to the code to apply changes to.
            changes: A list of changes.

        Returns:
            A string corresponding to the changed code.
        """


@attrs.define(eq=False, repr=False)
class BaseCodeRemovalChange(BaseCodeChange, ABC):
    """
    Base class for code changes based purely on removal of code.
    """


@attrs.define(eq=True, hash=True)
class SimpleAstNodeRef:
    """
    A serializable and comparable reference to a native Python AST node.
    This relies on the native ast.walk function to always return nodes in the same order. It obviously also relies on
    the parse tree not changing.
    """
    node_type: str
    index: int

    @classmethod
    def get_refs(cls, code: Union[str, ast.AST]) -> Dict['SimpleAstNodeRef', ast.AST]:
        """
        Returns a dictionary mapping node-refs to nodes given a code or a native AST instance.

        Args:
            code: A string or a native AST root.

        Returns:
            A dictionary mapping node-refs to AST nodes.
        """
        if isinstance(code, str):
            code_ast: ast.AST = ast.parse(code)
        else:  # code is already ast.AST
            code_ast: ast.AST = code

        type_ctr = collections.Counter()
        ref_dict: Dict[SimpleAstNodeRef, ast.AST] = {}

        for node in ast.walk(code_ast):
            if isinstance(node, ast.AST):
                node_type = type(node).__name__
                ctr = type_ctr[node_type]
                type_ctr[node_type] += 1
                ref_dict[SimpleAstNodeRef(node_type, ctr)] = node

        return ref_dict


@attrs.define(eq=False, repr=False)
class SimpleAstRemovalChange(BaseCodeRemovalChange):
    """
    A simple removal-based change class that removes nodes from an AST to apply the code change.
    """
    #  Node-refs to remove as part of this change
    node_refs: List[SimpleAstNodeRef]

    #  The *logical* list of changes that should also be applied if this change is being applied.
    #  As an example suppose the parent change is removal of a function call, while a child change can be
    #  the removal of a single keyword argument, along with any statements solely involved in the computation
    #  of the argument.
    children: List['SimpleAstRemovalChange']

    @classmethod
    def apply_changes(cls, code: str, changes: List['BaseCodeRemovalChange']) -> str:
        """
        Applies the code-change to the given code and returns the result.

        It gathers all the node-refs across the changes and their children and removes them from the AST generated for
        the code.

        NOTE: As of now, all the changes must be `SimpleAstRemovalChange` instances.

        Args:
            code: A string corresponding to the code to apply changes to.
            changes: A list of `SimpleAstRemovalChange` instances.

        Returns:
            A string corresponding to the changed code.
        """
        if any(not isinstance(c, SimpleAstRemovalChange) for c in changes):
            raise TypeError(f"Cannot handle heterogeneous change-types in {cls.__name__}")

        code_ast = ast.parse(code)
        ref_dict: Dict[SimpleAstNodeRef, ast.AST] = SimpleAstNodeRef.get_refs(code_ast)
        #  Need to include the children in the changes as well.
        all_changes = changes[:]
        for c in changes:
            all_changes.extend(c.children)

        all_changes = list(set(all_changes))

        #  Collect all the node-refs of the changes.
        to_remove: List[ast.AST] = sum(
            ([ref_dict[r] for r in change.node_refs] for change in all_changes),
            []
        )

        new_code = codeutils.unparse_native_ast(astutils.remove_nodes_from_native_ast(code_ast, to_remove))
        return new_code


@attrs.define(eq=True, hash=True)
class SimpleAstLibNodeRef:
    """
    A serializable and comparable reference to a astlib Python AST node.
    This relies on the astlib.walk function to always return nodes in the same order. It obviously also relies on
    the parse tree not changing.
    """
    node_type: str
    index: int

    @classmethod
    def get_refs(cls, code: Union[str, astlib.AstNode]) -> Dict['SimpleAstLibNodeRef', astlib.AstNode]:
        """
        Returns a dictionary mapping node-refs to nodes given a code or a astlib AST instance.

        Args:
            code: A string or a astlib AST root.

        Returns:
            A dictionary mapping node-refs to astlib AST nodes.
        """
        if isinstance(code, str):
            code_ast: astlib.AstNode = astlib.parse(code)
        else:  # code is already ast.AST
            code_ast: astlib.AstNode = code

        type_ctr = collections.Counter()
        ref_dict: Dict[SimpleAstLibNodeRef, astlib.AstNode] = {}

        for node in astlib.walk(code_ast):
            node_type = type(node).__name__
            ctr = type_ctr[node_type]
            type_ctr[node_type] += 1
            ref_dict[SimpleAstLibNodeRef(node_type, ctr)] = node

        return ref_dict


@attrs.define(eq=False, repr=False)
class SimpleAstLibRemovalChange(BaseCodeRemovalChange):
    """
    A simple removal-based change class that removes nodes from an AST to apply the code change.
    Similar to SimpleAstRemovalChange with the primary difference being the use of astlib instead of the native ast
    module.
    """
    #  Node-refs to remove as part of this change
    node_refs: List[SimpleAstLibNodeRef]

    #  The *logical* list of changes that should also be applied if this change is being applied.
    #  As an example suppose the parent change is removal of a function call, while a child change can be
    #  the removal of a single keyword argument, along with any statements solely involved in the computation
    #  of the argument.
    children: List['SimpleAstLibRemovalChange']

    @classmethod
    def apply_changes(cls, code: str, changes: List['BaseCodeRemovalChange']) -> str:
        """
        Applies the code-change to the given code and returns the result.

        It gathers all the node-refs across the changes and their children and removes them from the AST generated for
        the code.

        NOTE: As of now, all the changes must be `SimpleAstLibRemovalChange` instances.

        Args:
            code: A string corresponding to the code to apply changes to.
            changes: A list of `SimpleAstLibRemovalChange` instances.

        Returns:
            A string corresponding to the changed code.
        """
        if any(not isinstance(c, SimpleAstLibRemovalChange) for c in changes):
            raise TypeError(f"Cannot handle heterogeneous change-types in {cls.__name__}")

        code_ast = astlib.parse(code)
        ref_dict: Dict[SimpleAstLibNodeRef, astlib.AstNode] = SimpleAstLibNodeRef.get_refs(code_ast)
        #  Need to include the children in the changes as well.
        all_changes = changes[:]
        for c in changes:
            all_changes.extend(c.children)

        all_changes = list(set(all_changes))

        #  Collect all the node-refs of the changes.
        to_remove: List[astlib.AstNode] = sum(
            ([ref_dict[r] for r in change.node_refs] for change in all_changes),
            []
        )

        new_code = codeutils.unparse_astlib_ast(astutils.remove_nodes_from_astlib_ast(code_ast, to_remove))
        return new_code
