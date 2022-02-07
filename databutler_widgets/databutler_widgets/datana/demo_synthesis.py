"""
Entry-point to synthesis.
This is just for demo purposes, this will eventually be integrated within databutler, not databutler_widgets
"""
import base64
import collections
import io
from typing import List, Union, Type, Dict, Any

import pandas as pd
import matplotlib as mpl
from IPython import display

from databutler_widgets.datana.widget import DatanaExampleWidget
from databutler_widgets.datana import demo_corpus
from databutler.datana.viz.utils import mpl_exec

mpl.use('Agg')


class _AllColumns:
    pass


def _serialize_fig(fig, tight: bool = True):
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, bbox_inches='tight', format='png')
    else:
        fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()


def _run_func_and_get_img_src_as_base64(code: str, func_name: str, pos_args: List[Any], kw_args: Dict[str, Any]) -> str:
    #  Currently assumes the functions run correctly
    fig = mpl_exec.run_viz_code_matplotlib_mp(code, func_name=func_name, pos_args=pos_args, kw_args=kw_args)
    png_bytes = _serialize_fig(fig, tight=True)
    b64_val = base64.b64encode(png_bytes).decode('utf-8')
    return f"data:image/png;base64,{b64_val}"


def _get_search_options(df: pd.DataFrame, columns: Union[List[str], Type[_AllColumns]]):
    if columns is _AllColumns or len(columns) != 1:
        raise NotImplementedError("Only handling single columns for now")

    #  Collect base/vanilla descriptions from the corpus. We will use these as the search options.
    #  They must have an entry for [COL0] in the text
    vanilla_desc_map: Dict[str, List[demo_corpus.CorpusItem]] = collections.defaultdict(list)
    for item in demo_corpus.CORPUS:
        if "[COL0]" in item.vanilla_desc:
            vanilla_desc_map[item.vanilla_desc].append(item)

    #  Replace instances of [COL0] with the actual column name
    vanilla_desc_map = {k.replace("[COL0]", columns[0]): v for k, v in vanilla_desc_map.items()}

    return vanilla_desc_map


def synthesize(df: pd.DataFrame, columns: Union[List[str], Type[_AllColumns]] = _AllColumns):
    if columns is _AllColumns or len(columns) != 1:
        raise NotImplementedError("Only handling single columns for now")

    search_options = _get_search_options(df, columns)
    widget = DatanaExampleWidget()

    #  Setup the search options
    def _update_search_options():
        widget.search_options = [
            {"id": str(idx), 'title': title}
            for idx, title in enumerate(search_options.keys())
        ]

    widget.callback_method(_update_search_options, "search_box_value")

    def _update_graphs():
        key = str(widget.search_box_value)
        options: List[demo_corpus.CorpusItem] = search_options[key]

        #  Run each function and get the output as b64
        graphs: List[Dict] = []
        for item in options:
            code = item.func.code_str
            pos_args = []
            kw_args = {"df": df, "col0": columns[0]}
            img_src = _run_func_and_get_img_src_as_base64(code, item.func.func_name, pos_args, kw_args)

            graphs.append({
                "id": item.uid,
                "addr": img_src,
            })

        widget.graphs_generated = graphs

    widget.callback_method(_update_graphs, "search_selected")

    display.display(widget)
