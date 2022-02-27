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
import matplotlib.pyplot as plt
from IPython import display

from databutler_widgets.datana.widget import DatanaExampleWidget
from databutler_widgets.datana import demo_corpus
from databutler.datana.viz.utils import mpl_exec

mpl.use('Agg')

LOADING_URL = ("https://www.pinclipart.com/picdir/middle/"
               "543-5431019_nail-polish-hand-nail-transparent-background-loading-icon.png")


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
    display.display(code)
    fig = mpl_exec.run_viz_code_matplotlib_mp(code, func_name=func_name, pos_args=pos_args, kw_args=kw_args)
    png_bytes = _serialize_fig(fig, tight=True)
    plt.close(fig)
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

    vanilla_desc_map["Plot a line-graph for [COL0]"] = []
    vanilla_desc_map["Plot missing values in [COL0] using a heatmap"] = []

    #  Replace instances of [COL0] with the actual column name
    vanilla_desc_map = {k.replace("[COL0]", columns[0]): v for k, v in vanilla_desc_map.items()}

    return vanilla_desc_map


def _get_uid_to_corpus_item_map() -> Dict[str, demo_corpus.CorpusItem]:
    return {item.uid: item for item in demo_corpus.CORPUS}


def synthesize(df: pd.DataFrame, columns: Union[List[str], Type[_AllColumns]] = _AllColumns,
               variant_mode: str = "options+code"):
    if columns is _AllColumns or len(columns) != 1:
        raise NotImplementedError("Only handling single columns for now")

    show_variant_options = variant_mode in {"options+code", "options"}
    show_code = variant_mode in {"options+code", "code"}

    search_options = _get_search_options(df, columns)
    uid_to_corpus_items: Dict[str, demo_corpus.CorpusItem] = _get_uid_to_corpus_item_map()
    widget = DatanaExampleWidget()

    #  Setup the search options
    def _update_search_options():
        widget.search_options = [
            {"id": str(idx), 'title': title}
            for idx, title in enumerate(search_options.keys())
        ]

    widget.callback_method(_update_search_options, "search_box_value")

    def _update_graphs():
        if not widget.search_selected:
            return

        key = str(widget.search_box_value)
        options: List[demo_corpus.CorpusItem] = search_options[key]

        #  Run each function and get the output as b64
        graphs: List[Dict] = []
        for item in options:
            code = item.func.code_str
            pos_args = []
            kw_args = {"df": df, "col0": columns[0]}
            img_src = _run_func_and_get_img_src_as_base64(code, item.func.func_name, pos_args, kw_args)
            display.display([v[1] for v in item.change_dict.values()])

            graphs.append({
                "id": item.uid,
                "addr": img_src,
                "code": code,
                "show_options": show_variant_options,
                "show_code": show_code,
                "variant_desc": [{
                    "id": k,
                    "desc": v[1]
                } for k, v in item.change_dict.items()],
            })

        widget.graphs_generated = graphs

    widget.callback_method(_update_graphs, "search_selected")

    def _update_highlighted_graph():
        change = widget.unchecked_mods_list
        cur_item = uid_to_corpus_items[widget.highlighted_graph["id"]]

        if change.get("change", None) == "cur_options":
            unchecked_change_ids: List[str] = list(map(str, widget.unchecked_mods_list["cur_options"]))
            new_code = cur_item.apply_changes(unchecked_change_ids)
        else:
            new_code = change["cur_code"]

        pos_args = []
        kw_args = {"df": df, "col0": columns[0]}
        widget.highlighted_graph = {
            "id": cur_item.uid,
            "addr": LOADING_URL,
            "code": "",
            "show_options": show_variant_options,
            "show_code": show_code,
            "variant_desc": widget.highlighted_graph['variant_desc']
        }

        img_src = _run_func_and_get_img_src_as_base64(new_code, cur_item.func.func_name, pos_args, kw_args)

        widget.highlighted_graph = {
            "id": cur_item.uid,
            "addr": img_src,
            "code": new_code,
            "show_options": show_variant_options,
            "show_code": show_code,
            "variant_desc": widget.highlighted_graph['variant_desc']
        }

    widget.callback_method(_update_highlighted_graph, "unchecked_mods_list")

    display.display(widget)
