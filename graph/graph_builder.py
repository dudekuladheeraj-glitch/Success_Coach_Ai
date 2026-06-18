from langgraph.graph import StateGraph

from graph.state import CoachState
from graph.nodes import chat_node

builder = StateGraph(CoachState)

builder.add_node(
    "chat",
    chat_node
)

builder.set_entry_point("chat")
builder.set_finish_point("chat")

graph = builder.compile()