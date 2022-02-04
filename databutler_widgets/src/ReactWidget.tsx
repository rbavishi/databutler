import React, { useState } from 'react';
import { WidgetModel } from '@jupyter-widgets/base';
import { useModelState, WidgetModelContext } from './hooks/widget-model';

interface WidgetProps {
  model: WidgetModel;
}
function getData(input: any) {
  return [{
    "id": "1",
    "title": "pie chart"
  },
  {
      "id": "2",
      "title": "bar chart"
  },
  {
      "id": "3",
      "title": "stacked chart"
  },
  {
      "id": "4",
      "title": "line chart"
  },
  {
      "id": "5",
      "title": "histogram"
  }]
}

function ReactWidget(props: WidgetProps) {
  const [input, setInput] = useModelState('value');
  const [select, setSelect] = useState(false);
  const [callbackDummy, setCallbackDummy] = useModelState('callback_dummy');

  const dropBar = getData(input).map((post) => (
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
        placeholder="Hello"
        onChange={(e) => setInput(e.target.value)}
      />
      <button onClick={(e) => {setSelect(true); setCallbackDummy(0)}}>Select</button>
      </>
      }
      {!select && input && dropBar}
      {select && <Selection input={input} />}
    </div>
  );
}

function Selection(input: any) {
  const [image, setImage] = useState(0);
  const graphs = getGraphs(input, setImage)
  return (
    <>
    <div>{graphs}</div>
    {image !== 0 && <DisplayGraph addr={image} />}
    </>
  )
}

function DisplayGraph({ addr }: {addr: any}) {
  return (
    <div style={{paddingTop: "30px", display: "flex", flexDirection: "row"}}>
    <img src={addr}
              style={{border: "1px solid black", width: "600px"}} />
    <div className="options">
      {getOptions(addr)}
    </div>

    </div>
  )
}

function getOptions(addr: any) {
  return [
  <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" />Lorem Ipsum</div>,
  <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" />Lorem Ipsum</div>,
  <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" />Lorem Ipsum</div>,
  <div><input type="checkbox" id="topping" name="topping" value="Lorem Ipsum" />Lorem Ipsum</div>
  ]
}

function getGraphs(input: any, setId: any) {
  return [
    ImageBox(image_addr, 1, setId),
    ImageBox(image_addr, 2, setId),
    ImageBox(image_addr, 3, setId)
  ]
}

function ImageBox(addr: any, id: any, setId: any) {
  return <img src={addr}
              id={"image" + id}
              style={{border: "1px solid black"}}
              onClick={(e) => {setId(addr)}}></img>
}

const image_addr = "https://chartio.com/assets/dfd59f/tutorials/charts/grouped-bar-charts/c1fde6017511bbef7ba9bb245a113c07f8ff32173a7c0d742a4e1eac1930a3c5/grouped-bar-example-1.png"

function withModelContext(Component: (props: WidgetProps) => JSX.Element) {
  return (props: WidgetProps) => (
    <WidgetModelContext.Provider value={props.model}>
      <Component {...props} />
    </WidgetModelContext.Provider>
  );
}

export default withModelContext(ReactWidget);
