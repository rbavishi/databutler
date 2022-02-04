#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Rohan Bavishi.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""
from typing import Callable

from ipywidgets import DOMWidget
from traitlets import Unicode, Int
from .._frontend import module_name, module_version


class DatanaExampleWidget(DOMWidget):
    """TODO: Add docstring here
    """
    _model_name = Unicode('DatanaExampleModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('DatanaExampleView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    # Your widget state goes here. Make sure to update the corresponding
    # JavaScript widget state (defaultModelProperties) in widget.ts
    value = Unicode('Jupyter-Datana').tag(sync=True)
    callback_dummy = Int(-1).tag(sync=True)

    def callback_method(self, my_func: Callable):
        def wrapped_callback(*args, **kwargs):
            if self.callback_dummy != -1:
                my_func()

        self.observe(wrapped_callback, 'callback_dummy')
