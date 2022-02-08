#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Rohan Bavishi.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""
from typing import Callable

from ipywidgets import DOMWidget
from traitlets import Unicode, Bool, List, Dict
from .._frontend import module_name, module_version

IMAGE_LINK = "https://chartio.com/assets/dfd59f/tutorials/charts/grouped-bar-charts/c1fde6017511bbef7ba9bb245a113c07f8ff32173a7c0d742a4e1eac1930a3c5/grouped-bar-example-1.png"


class DatanaExampleWidget(DOMWidget):
    _model_name = Unicode('DatanaExampleModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('DatanaExampleView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    # Your widget state goes here. Make sure to update the corresponding
    # JavaScript widget state (defaultModelProperties) in widget.ts
    search_box_value = Unicode('').tag(sync=True)
    search_options = List([]).tag(sync=True)
    search_selected = Bool(False).tag(sync=True)

    graphs_generated = List([]).tag(sync=True)

    highlighted_graph = Dict({}).tag(sync=True)
    unchecked_mods_list = List([]).tag(sync=True)

    def __init__(self):
        super().__init__()

    def callback_method(self, callback_fn: Callable, callback_var: str):
        # the callback function never takes any arguments
        def wrapped_callback(*args, **kwargs):
            callback_fn()
        self.observe(wrapped_callback, callback_var)


