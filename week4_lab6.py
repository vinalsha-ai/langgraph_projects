import os
import sqlite3
import requests
import gradio as gr
from typing import Annotated
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv(override=True)

### Let's create tools

serper = GoogleSerperAPIWrapper()

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you want to do an online search"
)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str):
    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": text
    }
    requests.post(
        pushover_url,
        data=payload
    )

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Useful for when you want to send a push notification"
)

tools_list = [tool_search, tool_push]

### ### [PHASE-1] Define the Graph

### [PHASE-1][STEP-1] Define the State class

# class State(BaseModel):

#     messages: Annotated[list, add_messages]

class State(TypedDict):

    messages: Annotated[list, add_messages]

### [PHASE-1][STEP-2] Start the Graph Builder

graph_builder = StateGraph(State)

### [PHASE-1][STEP-3] Create a Node

llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools=tools_list)

def chatbot_node(old_state: State) -> State:
    # messages = old_state.messages  # BaseModel
    messages = old_state.get("messages")  # TypedDict
    response = llm_with_tools.invoke(messages)
    new_messages = [response]
    # new_state = State(messages=new_messages)  # BaseModel
    new_state = {"messages": new_messages}  # TypedDict
    print(new_state)
    return new_state

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools=tools_list))

### [PHASE-1][STEP-4] Create Edges

graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

### [PHASE-1][STEP-5] Compile the Graph

# memory = MemorySaver()

db_path = "memory.db"
conn = sqlite3.connect(
    db_path,
    check_same_thread=False
)
memory = SqliteSaver(conn)  # SQL Memory

graph = graph_builder.compile(checkpointer=memory)

### ### [PHASE-2] Execution

config = {
    "configurable": {
        "thread_id": "1"
    }
}

def chat(user_query: str, history):
    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]
    # state = State(messages=messages)  # BaseModel
    state = {"messages": messages}  # TypedDict
    result = graph.invoke(state, config=config)
    print(result)
    # print(f"\ngraph.get_state(config): {graph.get_state(config)}\n")
    # print(f"\nlist(graph.get_state_history(config)): {list(graph.get_state_history(config))}\n")
    return result["messages"][-1].content

gr.ChatInterface(fn=chat).launch()
