import json
from dataclasses import dataclass
from typing import Union, Sequence, Optional

import libcst as cst
from libcst import CSTVisitorT
from libcst._add_slots import add_slots
from libcst._nodes.internal import (
    CodegenState,
    visit_body_sequence,
    visit_required,
    visit_sequence,
)

MAGICS = {
    "automagic": "AutoMagics",
    "autocall": "AutoMagics",
    "alias_magic": "BasicMagics",
    "lsmagic": "BasicMagics",
    "magic": "BasicMagics",
    "page": "BasicMagics",
    "pprint": "BasicMagics",
    "colors": "BasicMagics",
    "xmode": "BasicMagics",
    "quickref": "BasicMagics",
    "doctest_mode": "BasicMagics",
    "gui": "BasicMagics",
    "precision": "BasicMagics",
    "notebook": "BasicMagics",
    "save": "CodeMagics",
    "pastebin": "CodeMagics",
    "loadpy": "CodeMagics",
    "load": "CodeMagics",
    "edit": "KernelMagics",
    "config": "ConfigMagics",
    "pdb": "ExecutionMagics",
    "tb": "ExecutionMagics",
    "run": "ExecutionMagics",
    "macro": "ExecutionMagics",
    "load_ext": "ExtensionMagics",
    "unload_ext": "ExtensionMagics",
    "reload_ext": "ExtensionMagics",
    "history": "HistoryMagics",
    "recall": "HistoryMagics",
    "rerun": "HistoryMagics",
    "logstart": "LoggingMagics",
    "logstop": "LoggingMagics",
    "logoff": "LoggingMagics",
    "logon": "LoggingMagics",
    "logstate": "LoggingMagics",
    "pinfo": "NamespaceMagics",
    "pinfo2": "NamespaceMagics",
    "pdef": "NamespaceMagics",
    "pdoc": "NamespaceMagics",
    "psource": "NamespaceMagics",
    "pfile": "NamespaceMagics",
    "psearch": "NamespaceMagics",
    "who_ls": "NamespaceMagics",
    "who": "NamespaceMagics",
    "whos": "NamespaceMagics",
    "reset": "NamespaceMagics",
    "reset_selective": "NamespaceMagics",
    "xdel": "NamespaceMagics",
    "alias": "OSMagics",
    "unalias": "OSMagics",
    "rehashx": "OSMagics",
    "pwd": "OSMagics",
    "cd": "OSMagics",
    "env": "OSMagics",
    "set_env": "OSMagics",
    "pushd": "OSMagics",
    "popd": "OSMagics",
    "dirs": "OSMagics",
    "dhist": "OSMagics",
    "sc": "OSMagics",
    "system": "OSMagics",
    "bookmark": "OSMagics",
    "pycat": "OSMagics",
    "pip": "PackagingMagics",
    "conda": "PackagingMagics",
    "matplotlib": "PylabMagics",
    "pylab": "PylabMagics",
    "killbgscripts": "ScriptMagics",
    "autoawait": "AsyncMagics",
    "ed": "Other",
    "hist": "Other",
    "rep": "Other",
    "clear": "KernelMagics",
    "less": "KernelMagics",
    "more": "KernelMagics",
    "man": "KernelMagics",
    "connect_info": "KernelMagics",
    "qtconsole": "KernelMagics",
    "autosave": "KernelMagics",
    "mkdir": "Other",
    "rmdir": "Other",
    "mv": "Other",
    "rm": "Other",
    "cp": "Other",
    "cat": "Other",
    "ls": "Other",
    "ll": "Other",
    "lf": "Other",
    "lk": "Other",
    "ldir": "Other",
    "lx": "Other",
    "store": "StoreMagics",
    "bigquery_stats": "Other",
    "js": "DisplayMagics",
    "javascript": "DisplayMagics",
    "latex": "DisplayMagics",
    "svg": "DisplayMagics",
    "html": "DisplayMagics",
    "markdown": "DisplayMagics",
    "capture": "ExecutionMagics",
    "!": "OSMagics",
    "writefile": "OSMagics",
    "script": "ScriptMagics",
    "sh": "Other",
    "bash": "Other",
    "perl": "Other",
    "ruby": "Other",
    "python": "Other",
    "python2": "Other",
    "python3": "Other",
    "pypy": "Other",
    "SVG": "Other",
    "HTML": "Other",
    "file": "Other",
    "bigquery": "Other",
}


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


def _remove_magics_from_cell_code(code: str) -> str:
    new_lines = []
    for line in code.split("\n"):
        line_s = line.lstrip()
        if line_s.startswith("%") or line_s.startswith("!") or line_s.startswith("?"):
            continue

        line_s_split = line_s.split()
        if len(line_s_split) > 0 and line_s_split[0] in MAGICS:
            #  We'll have to try to parse the statement as you can still have variables like cp
            #  We do not want to discard assignments such as "cp = 10"
            if not (len(line_s_split) > 2 and line_s_split[1].endswith("=")):
                #  TODO: This is finnicky. Need a robust way.
                try:
                    cst.parse_statement(line_s)
                except cst.ParserSyntaxError:
                    continue

        new_lines.append(line)

    return "\n".join(new_lines)


def parse_ipynb(src: Union[str, dict], python_version: str = None):
    if python_version is not None:
        config = cst.PartialParserConfig(
            encoding="utf-8", python_version=python_version
        )
    else:
        config = cst.PartialParserConfig(encoding="utf-8")

    if isinstance(src, str):
        src: dict = json.loads(src)

    cell_nodes = []

    for cell in src["cells"]:
        cell_src = "".join(cell["source"])
        cell_type = cell["cell_type"]
        if cell_type == "code":
            try:
                module = cst.parse_module(cell_src, config)
            except cst.ParserSyntaxError:
                #  Try after removing magics
                module = cst.parse_module(
                    _remove_magics_from_cell_code(cell_src), config
                )

            body = NotebookCellBody(body=module.body)
            cell_nodes.append(
                NotebookCell(
                    body=body, markdown=None, header=module.header, footer=module.footer
                )
            )

        elif cell_type == "markdown":
            body = NotebookCellBody(body=[])
            cell_nodes.append(NotebookCell(body=body, markdown=cell_src))

    return cst.Module(body=cell_nodes)
