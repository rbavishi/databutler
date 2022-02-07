// Copyright (c) Rohan Bavishi
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';
import ReactWidget from "./ReactWidget"
import React from 'react';
import ReactDOM from 'react-dom';

import { MODULE_NAME, MODULE_VERSION } from './version';

// Import the CSS
import '../css/widget.css';

// Your widget state goes here. Make sure to update the corresponding
// Python state in widget.py
const defaultModelProperties = {
  search_box_value: '',
  search_selected: false,
  search_options: [],
  graphs_generated: [],
  highlighted_graph: {},
  mods_list: []
}

export type WidgetModelState = typeof defaultModelProperties

export class DatanaExampleModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: DatanaExampleModel.model_name,
      _model_module: DatanaExampleModel.model_module,
      _model_module_version: DatanaExampleModel.model_module_version,
      _view_name: DatanaExampleModel.view_name,
      _view_module: DatanaExampleModel.view_module,
      _view_module_version: DatanaExampleModel.view_module_version,
      ...defaultModelProperties
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    // Add any extra serializers here
  };

  static model_name = 'DatanaExampleModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'DatanaExampleView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class DatanaExampleView extends DOMWidgetView {
  render() {
    this.el.classList.add('custom-widget');

    const component = React.createElement(ReactWidget, {
      model: this.model,
    });
    ReactDOM.render(component, this.el);
  }
}

export class VizsmithExampleModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: VizsmithExampleModel.model_name,
      _model_module: VizsmithExampleModel.model_module,
      _model_module_version: VizsmithExampleModel.model_module_version,
      _view_name: VizsmithExampleModel.view_name,
      _view_module: VizsmithExampleModel.view_module,
      _view_module_version: VizsmithExampleModel.view_module_version,
      ...defaultModelProperties
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    // Add any extra serializers here
  };

  static model_name = 'VizsmithExampleModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'VizsmithExampleView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class VizsmithExampleView extends DOMWidgetView {
  render() {
    this.el.classList.add('custom-widget');

    const component = React.createElement(ReactWidget, {
      model: this.model,
    });
    ReactDOM.render(component, this.el);
  }
}
