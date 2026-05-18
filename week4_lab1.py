# Imports: These are in order of which they are used

from dotenv import load_dotenv
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import random
import gradio as gr

# Constants

nouns = ["Cabbages", "Unicorns", "Toasters", "Penguins", "Bananas", "Zombies", "Rainbows", "Eels", "Pickles", "Muffins"]

adjectives = ["outrageous", "smelly", "pedantic", "existential", "moody", "sparkly", "untrustworthy", "sarcastic", "squishy", "haunted"]

# Load the environment variables

load_dotenv(override=True)

# 'TypeHinting' is a feature in python that let's us specify the type of 
# something, e.g. "def shout(text: str) -> None". 'str' is the type of 
# the 'text' parameter to the shout function, 'None' is the return type
# of the shout function. 

# 'Annotated' is another way to add extra information that someone else
# might find useful. And LangGraph needs us to use this feature when we
# define the State object. It wants us to tell which function it should
# call to update the State with a new value. This function is called a 
# "reducer". LangGraph provides a default reducer called add_messages, 
# which takes care of the most common case. And hopefully that explains
# why the State looks like this.


def shout(text: Annotated[str, "Somthing to be shouted"]) -> str:
    shout_text = text.upper()
    print(shout_text)
    return shout_text


### PHASE-1: Steps 1-5 ###

### Step-1: Define the State object ###

# We can use any Python object to define the State object, but it is 
# most common to use a TypedDict or a Pydantic BaseModel as one. In 
# this example, let's use Pydantic BaseModel.


class State(BaseModel):

    messages: Annotated[list, add_messages]


### Step-2: Start the Graph Builder with this State class ###

# We start the Graph Builder, then add nodes and edges to it and
# finally compile the Graph.

graph_builder = StateGraph(State)

### Step-3: Create a Node ###

# A node can be any python function that implements the Agent logic. 
# Nodes do the work. State object is immutable. So, node takes the 
# current State as input and returns a new State. The reducer that 
# we set before gets automatically called to combine this response 
# with previous responses.


def our_first_node(old_state: State) -> State:
    random_nouns = random.choice(nouns)
    random_adjectives = random.choice(adjectives)
    reply = f"{random_nouns} are {random_adjectives}"
    messages = [
        {
            "role": "assistant",
            "content": reply
        }
    ]
    new_state = State(messages=messages)
    return new_state


graph_builder.add_node("first_node", our_first_node)

### Step-4: Create Edges ###

# Edges are the lines connecting the nodes. They are also python functions
# that tell what to execute next, I mean what node to execute next in the 
# flow. They can be fixed or conditional. I do not want to belabour this.

graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", END)

### Step-5: Complile the Graph ###

graph = graph_builder.compile()

### PHASE-2 ###


def chat(user_input: str, history):
    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]
    state = State(messages=messages)
    result = graph.invoke(state)
    print(result)
    return result["messages"][-1].content


gr.ChatInterface(fn=chat).launch()
