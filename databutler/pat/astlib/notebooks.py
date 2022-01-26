import json
from dataclasses import dataclass
from typing import Union, Sequence, Optional

import libcst as cst
from libcst import CSTVisitorT
from libcst._add_slots import add_slots
from libcst._nodes.internal import CodegenState, visit_body_sequence, visit_required, visit_sequence


@add_slots
@dataclass(frozen=True)
class Notebook(cst.Module):
    pass


@add_slots
@dataclass(frozen=True)
class NotebookCellBody(cst.BaseSuite):
    body: Sequence[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]]

    def _visit_and_replace_children(self, visitor: CSTVisitorT) -> cst.CSTNode:
        return NotebookCellBody(
            body=visit_body_sequence(self, "body", self.body, visitor),
        )
        pass

    def _codegen_impl(self, state: CodegenState) -> None:
        for stmt in self.body:
            stmt._codegen(state)


@add_slots
@dataclass(frozen=True)
class NotebookCell(cst.BaseCompoundStatement):
    markdown: Optional[str]
    body: NotebookCellBody
    header: Sequence[cst.EmptyLine] = ()
    footer: Sequence[cst.EmptyLine] = ()

    def _visit_and_replace_children(self, visitor: CSTVisitorT) -> cst.CSTNode:
        return NotebookCell(
            body=visit_required(self, "body", self.body, visitor),
            markdown=self.markdown,
            header=visit_sequence(self, "header", self.header, visitor),
            footer=visit_sequence(self, "footer", self.footer, visitor),
        )

    def _codegen_impl(self, state: CodegenState) -> None:
        self.body._codegen(state)


def parse_ipynb(src: Union[str, dict], python_version: str = None):
    if python_version is not None:
        config = cst.PartialParserConfig(encoding="utf-8", python_version=python_version)
    else:
        config = cst.PartialParserConfig(encoding="utf-8")

    if isinstance(src, str):
        src: dict = json.loads(src)

    cell_nodes = []

    for cell in src['cells']:
        cell_src = "".join(cell['source'])
        cell_type = cell['cell_type']
        if cell_type == 'code':
            module = cst.parse_module(cell_src, config)
            body = NotebookCellBody(body=module.body)
            cell_nodes.append(NotebookCell(body=body, markdown=None, header=module.header, footer=module.footer))

        elif cell_type == 'markdown':
            body = NotebookCellBody(body=[])
            cell_nodes.append(NotebookCell(body=body, markdown=cell_src))

    return cst.Module(body=cell_nodes)
