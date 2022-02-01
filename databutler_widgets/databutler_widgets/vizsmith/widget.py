#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Rohan Bavishi.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget
from traitlets import Unicode
from .._frontend import module_name, module_version


class VizsmithExampleWidget(DOMWidget):
    """TODO: Add docstring here
    """
    _model_name = Unicode('VizsmithExampleModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('VizsmithExampleView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    # Your widget state goes here. Make sure to update the corresponding
    # JavaScript widget state (defaultModelProperties) in widget.ts
    value = Unicode('Jupyter-Vizsmith').tag(sync=True)
