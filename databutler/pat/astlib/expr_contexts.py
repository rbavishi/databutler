from enum import Enum, auto
import libcst as cst
from typing import Sequence, Optional

from libcst import BatchableMetadataProvider


class ExpressionContext(Enum):
    """
    Used in :class:`ExpressionContextProvider` to represent context of a variable reference.
    Copied from the library.
    """

    #: Load the value of a variable reference.
    #:
    #: >>> libcst.MetadataWrapper(libcst.parse_module("a")).resolve(libcst.ExpressionContextProvider)
    #: mappingproxy({Name(
    #:                   value='a',
    #:                   lpar=[],
    #:                   rpar=[],
    #:               ): <ExpressionContext.LOAD: 1>})
    LOAD = auto()

    #: Store a value to a variable reference by :class:`~libcst.Assign` (``=``),
    #: :class:`~libcst.AugAssign` (e.g. ``+=``, ``-=``, etc), or
    #: :class:`~libcst.AnnAssign`.
    #:
    #: >>> libcst.MetadataWrapper(libcst.parse_module("a = b")).resolve(libcst.ExpressionContextProvider)
    #: mappingproxy({Name(
    #:               value='a',
    #:               lpar=[],
    #:               rpar=[],
    #:           ): <ExpressionContext.STORE: 2>, Name(
    #:               value='b',
    #:               lpar=[],
    #:               rpar=[],
    #:           ): <ExpressionContext.LOAD: 1>})
    STORE = auto()

    #: Delete value of a variable reference by ``del``.
    #:
    #: >>> libcst.MetadataWrapper(libcst.parse_module("del a")).resolve(libcst.ExpressionContextProvider)
    #: mappingproxy({Name(
    #:                   value='a',
    #:                   lpar=[],
    #:                   rpar=[],
    #:               ): < ExpressionContext.DEL: 3 >})
    DEL = auto()


class ExpressionContextVisitor(cst.CSTVisitor):
    """
    Copied from the library with modifications (corrections)
    """

    def __init__(
            self, provider: "ExpressionContextProvider", context: ExpressionContext
    ) -> None:
        self.provider = provider
        self.context = context

    def visit_Assign(self, node: cst.Assign) -> bool:
        for target in node.targets:
            target.visit(
                ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
            )
        node.value.visit(self)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        node.target.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.annotation.visit(self)
        value = node.value
        if value:
            value.visit(self)
        return False

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        node.target.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.value.visit(self)
        return False

    def visit_Name(self, node: cst.Name) -> bool:
        self.provider.set_metadata(node, self.context)
        return False

    def visit_AsName(self, node: cst.AsName) -> Optional[bool]:
        node.name.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        return False

    def visit_CompFor(self, node: cst.CompFor) -> bool:
        node.target.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.iter.visit(self)
        for i in node.ifs:
            i.visit(self)
        inner_for_in = node.inner_for_in
        if inner_for_in:
            inner_for_in.visit(self)
        return False

    def visit_For(self, node: cst.For) -> bool:
        node.target.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.iter.visit(self)
        node.body.visit(self)
        orelse = node.orelse
        if orelse:
            orelse.visit(self)
        return False

    def visit_Del(self, node: cst.Del) -> bool:
        node.target.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.DEL)
        )
        return False

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        self.provider.set_metadata(node, self.context)
        node.value.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.LOAD)
        )
        # don't visit attr (Name), so attr has no context
        return False

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        self.provider.set_metadata(node, self.context)
        node.value.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.LOAD)
        )
        n_slice = node.slice
        if isinstance(n_slice, Sequence):
            for sli in n_slice:
                sli.visit(
                    ExpressionContextVisitor(self.provider, ExpressionContext.LOAD)
                )
        else:
            n_slice.visit(ExpressionContextVisitor(self.provider, ExpressionContext.LOAD))
        return False

    def visit_Tuple(self, node: cst.Tuple) -> Optional[bool]:
        self.provider.set_metadata(node, self.context)

    def visit_List(self, node: cst.List) -> Optional[bool]:
        self.provider.set_metadata(node, self.context)

    def visit_StarredElement(self, node: cst.StarredElement) -> Optional[bool]:
        self.provider.set_metadata(node, self.context)

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        node.name.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.body.visit(self)
        for base in node.bases:
            base.visit(self)
        for keyword in node.keywords:
            keyword.visit(self)
        for decorator in node.decorators:
            decorator.visit(self)
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        node.name.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        node.params.visit(self)
        node.body.visit(self)
        for decorator in node.decorators:
            decorator.visit(self)
        returns = node.returns
        if returns:
            returns.visit(self)
        return False

    def visit_Param(self, node: cst.Param) -> Optional[bool]:
        node.name.visit(
            ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
        )
        annotation = node.annotation
        if annotation:
            annotation.visit(self)
        default = node.default
        if default:
            default.visit(self)
        return False

    def visit_NameItem(self, node: cst.NameItem) -> bool:
        return False

    def visit_Arg(self, node: cst.Arg) -> bool:
        self.provider.set_metadata(node, self.context)
        node.value.visit(self)

        if node.keyword:
            node.keyword.visit(
                ExpressionContextVisitor(self.provider, ExpressionContext.STORE)
            )

        return False

    def visit_ImportAlias(self, node: cst.ImportAlias) -> bool:
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        if isinstance(node.names, cst.ImportStar):
            return False
        else:
            for name in node.names:
                name.visit(self)

        return False


class ExpressionContextProvider(BatchableMetadataProvider[Optional[ExpressionContext]]):
    """
    Copied from the library.
    """

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        node.visit(ExpressionContextVisitor(self, ExpressionContext.LOAD))
