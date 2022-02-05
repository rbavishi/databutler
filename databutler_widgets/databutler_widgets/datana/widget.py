#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Rohan Bavishi.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""
from typing import Callable

from ipywidgets import DOMWidget
from traitlets import Unicode, Bool, List
from .._frontend import module_name, module_version


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

    def __init__(self):
        super().__init__()
        self.callback_method(lambda: self.update_search_options(), 'search_box_value')

    def update_search_options(self):
        # hard coded update -> change to trie structure later
        self.search_options = [
            {'id': '1', 'title': 'pie chart'},
            {'id': '2', 'title': 'bar chart'},
            {'id': '3', 'title': 'histogram'},
            {'id': '4', 'title': self.search_box_value},
        ]

    def callback_method(self, callback_fn: Callable, callback_var: str):
        # the callback function never takes any arguments
        def wrapped_callback(*args, **kwargs):
            callback_fn()
        self.observe(wrapped_callback, callback_var)


