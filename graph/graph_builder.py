# from langgraph.graph import END, StateGraph

# from graph.nodes import chat_node, classify_intent, fetch_student_data
# from graph.state import CoachState


# builder = StateGraph(CoachState)

# builder.add_node("classify", classify_intent)
# builder.add_node("fetch_data", fetch_student_data)
# builder.add_node("chat", chat_node)

# builder.set_entry_point("classify")

# builder.add_edge("classify", "fetch_data")
# builder.add_edge("fetch_data", "chat")
# builder.add_edge("chat", END)

# graph = builder.compile()

from langgraph.graph import END, StateGraph

from graph.nodes import chat_node, classify_intent, fetch_student_data, knowledge_base_node
from graph.state import CoachState


def route_after_fetch(state: dict) -> str:
    """
    After fetching student data, route to:
    - 'knowledge_base'  if the question is about the learning platform
    - 'chat'            for everything else (academics, exams, overview, general)
    """
    intent = state.get("intent", "general")
    if intent == "knowledge_base":
        return "knowledge_base"
    return "chat"


builder = StateGraph(CoachState)

builder.add_node("classify",        classify_intent)
builder.add_node("fetch_data",      fetch_student_data)
builder.add_node("knowledge_base",  knowledge_base_node)
builder.add_node("chat",            chat_node)

builder.set_entry_point("classify")

builder.add_edge("classify", "fetch_data")

# Conditional branch after fetching data
builder.add_conditional_edges(
    "fetch_data",
    route_after_fetch,
    {
        "knowledge_base": "knowledge_base",
        "chat":           "chat",
    },
)

builder.add_edge("knowledge_base", END)
builder.add_edge("chat",           END)

graph = builder.compile()
