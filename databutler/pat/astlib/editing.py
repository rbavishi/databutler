from typing import Union, Dict, Set, Optional, Collection
import libcst as cst


class ChildReplacementTransformer(cst.CSTTransformer):
    def __init__(self, replacements: Dict[cst.CSTNode, cst.CSTNode]) -> None:
        self.replacements = replacements
        self.output_mapping: Dict[cst.CSTNode, cst.CSTNode] = {}

    def on_visit(self, node: cst.CSTNode) -> bool:
        # If the node is one we are about to replace, we shouldn't
        # recurse down it, that would be a waste of time.
        return node not in self.replacements

    def on_leave(self,
                 original_node: cst.CSTNode,
                 updated_node: cst.CSTNode) -> Union[cst.CSTNode, cst.RemovalSentinel]:
        if original_node in self.replacements:
            updated_node = self.replacements[original_node]

        self.output_mapping[original_node] = updated_node
        return updated_node


class NodeRemovalTransformer(cst.CSTTransformer):
    def __init__(self, to_remove: Collection[cst.CSTNode]) -> None:
        self._to_remove = set(to_remove)

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode
                 ) -> Union[cst.CSTNode, cst.RemovalSentinel]:

        if original_node in self._to_remove:
            return cst.RemoveFromParent()
        else:
            return updated_node


class StmtRemovalAndSimplificationTransformer(cst.CSTTransformer):
    def __init__(self,
                 to_remove: Optional[Set[cst.BaseSmallStatement]],
                 to_retain: Optional[Set[cst.BaseSmallStatement]]) -> None:
        self.to_remove = to_remove
        self.to_retain = to_retain

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, cst.BaseSmallStatement):
            if self.to_remove is not None and node in self.to_remove:
                return False
            if self.to_retain is not None and node not in self.to_retain:
                return False
        return True

    def on_leave(
        self, original_node: cst.CSTNodeT, updated_node: cst.CSTNodeT
    ) -> Union[cst.CSTNodeT, cst.RemovalSentinel]:
        if isinstance(original_node, cst.BaseSmallStatement):
            if self.to_remove is not None and original_node in self.to_remove:
                return cst.RemoveFromParent()
            if self.to_retain is not None and original_node not in self.to_retain:
                return cst.RemoveFromParent()

        if isinstance(updated_node, cst.BaseCompoundStatement):
            if len(updated_node.body.body) == 0:
                if isinstance(updated_node, (cst.For, cst.While)):
                    if updated_node.orelse is None:
                        return cst.RemoveFromParent()
                elif isinstance(updated_node, cst.If):
                    if updated_node.orelse is None:
                        return cst.RemoveFromParent()
                elif isinstance(updated_node, cst.Try):
                    if len(updated_node.handlers) == 0 and \
                            updated_node.orelse is None and \
                            updated_node.finalbody is None:
                        return cst.RemoveFromParent()

                else:
                    return cst.RemoveFromParent()

        if isinstance(updated_node, (cst.Else, cst.Finally)):
            if len(updated_node.body.body) == 0:
                return cst.RemoveFromParent()

        if isinstance(updated_node, cst.BaseExpression):
            return original_node

        # print("RETURNING", cst.Module([]).code_for_node(updated_node).strip())
        return updated_node
