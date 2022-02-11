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
  const [myLocal, setMyLocal] = useState(false);
  const [displayDropBar, setDisplayDropBar] = useState(false);

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
      <>
      <input
        type="text"
        value={input}
        placeholder="Enter your visualization"
        onChange={(e) => {setInput(e.target.value); setDisplayDropBar(true);}}
      />
      <button onClick={(e) => {
          setSelect(false); setSelect(true); setDisplayDropBar(false);}
      }>Select</button>
      </>
      }
      {displayDropBar && input && dropBar}
      {select && <Selection input={input} />}
    </div>
  );
}

function Selection(input: any) {
  const [graphs, setGraphs] = useModelState('graphs_generated')
  const [highlightedGraph, setHighLightedGraph] = useModelState('highlighted_graph');

  return (
    <>
    <div>{formatGraphs(graphs, setHighLightedGraph)}</div>
    {JSON.stringify(highlightedGraph) !== '{}' && <DisplayGraph graph={highlightedGraph}/>}
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

const Console = prop => (
  console[Object.keys(prop)[0]](...Object.values(prop))
  ,null // ➜ React components must return something
)

function ImageBox({ graph, setHighlightedGraph } ) {
  return <img src={graph.addr}
              id={"image" + graph.id}
              style={{border: "1px solid black"}}
              onClick={(e) => {setHighlightedGraph(graph)}}></img>
}

function DisplayGraph({graph}) {
  const [modsList, setModsList] = useModelState('unchecked_mods_list');
  return (
    <div style={{paddingTop: "30px", display: "flex", flexDirection: "row"}}>
      <img src={graph.addr}
                style={{border: "1px solid black", width: "600px"}} />
      <div className="options">
        {formatOptions(graph, modsList, setModsList)}
      </div>
    </div>
  )
}

function formatOptions(graph: any, lst: any, setModsList) {
  return graph.variant_desc.map((mod: any) => (
    <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" checked={!(lst.indexOf(mod.id) > -1)}
                onChange={(e) => {
                    if (!e.target.checked) {
                        let lstCopy = [...lst];
                        lstCopy.push(mod.id);
                        setModsList(lstCopy);
                    }
                    else {
                        let lstCopy = [... lst];
                        if (lstCopy.indexOf(mod.id) > -1) {
                            lstCopy = lstCopy.filter((elem) => elem !== mod.id);
                            setModsList(lstCopy);
                        }
                    }
                }
                }/>{mod.desc}</div>
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
