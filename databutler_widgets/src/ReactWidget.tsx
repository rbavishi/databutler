import React, { useState } from 'react';
import { WidgetModel } from '@jupyter-widgets/base';
import { useModel, useModelState, WidgetModelContext } from './hooks/widget-model';

interface WidgetProps {
  model: WidgetModel;
}

function ReactWidget(props: WidgetProps) {
  const [input, setInput] = useModelState('search_box_value');
  const [searchOptions, _] = useModelState('search_options');
  const [select, setSelect] = useModelState('search_selected');

  const dropBar = searchOptions.map((post) => (
    <div key={post.id}>
      <p
        style={{padding: "10px", margin: "0"}}
        onClick={(e) => setInput(post.title)}
        onMouseEnter={(e: any) => {
            e.target.style.background = "lightgrey"
            e.target.style.cursor = "pointer"
          }
        }
        onMouseLeave={(e: any) => e.target.style.background = "transparent"}>
            {post.title}
      </p>
    </div>
  ))

  return (
    <div className="Widget">
    {
      !select &&
      <>
      <input
        type="text"
        value={input}
        placeholder="Enter your visualization"
        onChange={(e) => setInput(e.target.value)}
      />
      <button onClick={(e) => {setSelect(true);}}>Select</button>
      </>
      }
      {!select && input && dropBar}
      {select && <Selection input={input} />}
    </div>
  );
}

function Selection(input: any) {
  const [graphs, setGraphs] = useModelState('graphs_generated')
  const [highlightedGraph, setHighLightedGraph] = useModelState('highlighted_graph');
  const [modsList, setModsList] = useModelState('mods_list')


  return (
    <>
    <div>{formatGraphs(graphs, setHighLightedGraph)}</div>
    {highlightedGraph !== {} && <DisplayGraph graph={highlightedGraph} modsList={modsList}/>}
    </>
  )
}

function formatGraphs(graphList: any, setHighLightedGraph: any) {
  let graphs = graphList.map((graph) =>
    <>
    <ImageBox graph={graph} setHighlightedGraph={setHighLightedGraph} />
    </>
  )
  return graphs
}

function ImageBox({ graph, setHighlightedGraph } ) {
  return <img src={graph.addr}
              id={"image" + graph.id}
              style={{border: "1px solid black"}}
              onClick={(e) => {setHighlightedGraph(graph)}}></img>
}

function DisplayGraph({ graph, modsList }) {
  return (
    <div style={{paddingTop: "30px", display: "flex", flexDirection: "row"}}>
      <img src={graph.addr}
                style={{border: "1px solid black", width: "600px"}} />
      <div className="options">
        {formatOptions(modsList)}
      </div>
    </div>
  )
}

function formatOptions(lst: []) {
  return lst.map((mod) => (
    <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" />{mod}</div>
  ))
}

function withModelContext(Component: (props: WidgetProps) => JSX.Element) {
  return (props: WidgetProps) => (
    <WidgetModelContext.Provider value={props.model}>
      <Component {...props} />
    </WidgetModelContext.Provider>
  );
}

export default withModelContext(ReactWidget);
