from langchain_core.messages import HumanMessage, SystemMessage

from llm.openai_client import get_llm


llm = get_llm()


def generate_session_summary(messages: list[dict]) -> str:
    """
    Generate a concise summary of what the student shared.
    """

    student_messages = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "user"
    ]

    if not student_messages:
        return "No meaningful student input captured."

    conversation_text = "\n".join(
        f"- {message}"
        for message in student_messages
    )

    prompt = f"""
Summarize this coaching session.

Capture:

1. Concerns
2. Goals
3. Commitments
4. Important facts shared by the student
5. Recommended actions

Keep the summary concise and useful for future coaching sessions.

Conversation:

{conversation_text}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert academic coaching memory summarizer."
            ),
            HumanMessage(content=prompt),
        ]
    )

    return response.content