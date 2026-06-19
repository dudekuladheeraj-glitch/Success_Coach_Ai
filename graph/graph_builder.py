from langgraph.graph import END, StateGraph

from graph.nodes import chat_node, classify_intent, fetch_student_data
from graph.state import CoachState


builder = StateGraph(CoachState)

builder.add_node("classify", classify_intent)
builder.add_node("fetch_data", fetch_student_data)
builder.add_node("chat", chat_node)

builder.set_entry_point("classify")

builder.add_edge("classify", "fetch_data")
builder.add_edge("fetch_data", "chat")
builder.add_edge("chat", END)

graph = builder.compile()
