from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import os
import requests
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import gradio as gr
from typing import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)

serper = GoogleSerperAPIWrapper()

# user_query = "What was the stock price of ORCL on 12th/December/1989?"
# user_query = "What was the stock price of ORCL on 12th/December/2017?"
# user_query = "What was the closing stock price of ORCL on 12th/December/2017? Give me just the price in dollars."
user_query = "What is the highest price ORCL stock has touched till date?"

# print(serper.run(user_query))

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Used for when we need to do an online search"
)

# print(tool_search.invoke(user_query))

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
    description="Useful for when we want to send a push notification"
)

# tool_push.invoke("I am currently on Week-4 working up the LangChain/LangGraph/LangSmith!")

### ### [PHASE-1] Define the Graph

### [PHASE-1][STEP-1]: Define the State class

class State(TypedDict):

    messages: Annotated[list, add_messages]

### [PHASE-1][STEP-2]: Start the Graph Builder

graph_builder = StateGraph(State)

### [PHASE-1][STEP-3]: Create a Node

tools = [tool_search, tool_push]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(old_state: State) -> State:
    messages = old_state.get("messages")
    response = llm_with_tools.invoke(messages)
    new_state = {"messages": [response]}
    print(new_state)
    return new_state

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

### [PHASE-1][STEP-4]: Create Edges

graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

### [PHASE-1][STEP-5]: Compile the Graph

graph = graph_builder.compile()

### ### [PHASE-2] Execution

def chat(user_query: str, history):
    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]
    state = {"messages": messages}
    result = graph.invoke(state)
    print(result)
    return result["messages"][-1].content

gr.ChatInterface(fn=chat).launch()



