from typing import TypedDict


class CoachState(TypedDict):
    message:         str
    student_id:      str
    intent:          str
    student_context: dict
    alerts:          list[str]
    kb_context:      str        # knowledge base chunks retrieved from ChromaDB
    response:        str
    chat_history:    list[dict]