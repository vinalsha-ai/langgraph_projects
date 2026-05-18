from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import gradio as gr

load_dotenv(override=True)

### ### [PHASE-1] Setup before running the Agent

### [PHASE-1][STEP-1]: Define the State class

class State(BaseModel):

    messages: Annotated[list, add_messages]

### [PHASE-1][STEP-2]: Start the Graph Builder

graph_builder = StateGraph(State)

### [PHASE-1][STEP-3]: Create a Node

llm = ChatGroq(model="openai/gpt-oss-120b")

def chatbot_node(old_state: State) -> State:
    messages = old_state.messages
    response = llm.invoke(messages)
    new_state = State(messages=[response])
    return new_state

graph_builder.add_node("chatbot", chatbot_node)

### [PHASE-1][STEP-4]: Create Edges

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

### [PHASE-1][STEP-5]: Compile the Graph

graph = graph_builder.compile()

### ### [PHASE-2] Runing the Agent

def chat(user_query: str, history):
    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]
    state = State(messages=messages)
    result = graph.invoke(state)
    print(f"\nResult: {result}\n")
    return result["messages"][-1].content

gr.ChatInterface(fn=chat).launch()

### Explanation

"""

This script sets up a conversational AI chatbot by combining three major tools: **LangGraph** (to build the AI's structural brain/workflow), **LangChain & Groq** (to handle the actual language model thinking), and **Gradio** (to instantly deploy a web-based chat interface).

Here is a breakdown of what happens in each phase, step-by-step.

---

## Phase 1: Building the Architecture

This phase acts like a blueprint. It designs how information moves inside your chatbot, but it doesn't run the chat loop yet.

### Step 1: Define the "State"

```python
class State(BaseModel):
    messages: Annotated[list, add_messages]

```

* **What it does:** LangGraph applications rely entirely on a central "State" object—think of it as the shared database or memory bank for your application.
* **Why `add_messages` matters:** Instead of completely wiping out your history every time a new message comes in, `add_messages` tells the system to append (add) new incoming user queries or AI responses to the existing list, automatically preserving chat history.

### Step 2 & 3: The Builder and the Node

```python
graph_builder = StateGraph(State)
llm = ChatGroq(model="openai/gpt-oss-120b")

def chatbot_node(old_state: State) -> State:
    messages = old_state.messages
    response = llm.invoke(messages)
    return State(messages=[response])

graph_builder.add_node("chatbot", chatbot_node)

```

* **Nodes are workers:** In LangGraph, nodes are just standard Python functions that execute logic. Here, your `chatbot_node` takes the current conversation memory (`old_state`), hands it over to the Groq LLM API, grabs the text answer, and returns a new state containing the response.

### Steps 4 & 5: Mapping Routes and Compiling

```python
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

```

* **Edges are pathways:** Edges control the execution flow. You explicitly tell LangGraph: "When the application boots up (`START`), go directly to the `chatbot` node. Once the chatbot node finishes processing, exit the graph (`END`)."
* **Compiling:** The `.compile()` command takes your blueprint (nodes, memory states, and routing edges), validates that there are no dead-ends or broken pathways, and turns it into an active executable application engine (`graph`).

---

## Phase 2: Interactivity and the UI

Now that the engine is ready, Phase 2 wires it up to a user interface.

### The Chat Function

```python
def chat(user_query: str, history):
    messages = [{"role": "user", "content": user_query}]
    state = State(messages=messages)
    result = graph.invoke(state)
    return result["messages"][-1].content

```

* This function acts as the translator between Gradio's UI front-end and your LangGraph back-end.
* It intercepts whatever the user types, wraps it up in the structured dictionary format LangGraph expects, and boots up your workflow graph with `.invoke()`.
* Once the graph completes its processing run, it extracts the very last message in the list (`result["messages"][-1].content`), which is the text written by the AI, and passes it back to the UI.

### Launching the Frontend

```python
gr.ChatInterface(fn=chat).launch()

```

* Gradio's `ChatInterface` handles all the heavy UI lifting automatically. It spins up a local web server, creates a clean web browser chat window with text inputs, submit buttons, and conversation bubbles, and maps your `chat` logic to it seamlessly.

"""