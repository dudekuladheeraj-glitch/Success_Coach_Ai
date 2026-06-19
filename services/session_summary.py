from langchain_core.messages import HumanMessage, SystemMessage

from llm.openai_client import get_llm


llm = get_llm()


def _student_text(messages: list[dict]) -> str:
    student_messages = [
        msg["content"] for msg in messages if msg.get("role") == "user"
    ]
    if not student_messages:
        return ""
    return "\n".join(f"- {message}" for message in student_messages)


def generate_session_summary(messages: list[dict]) -> str:
    """
    Generate a concise NARRATIVE summary of the session — what was discussed,
    decided, and committed to. This is the "briefing" memory type, used when
    a coach asks "what happened in our last session" / "catch me up".
    """

    conversation_text = _student_text(messages)

    if not conversation_text:
        return "No meaningful student input captured."

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


def extract_facts(messages: list[dict]) -> str:
    """
    Generate a short bullet list of DURABLE FACTS about the student —
    things that should shape how the AI talks to them in future sessions,
    independent of any one conversation. This is the "factual memory" type.

    Examples of what belongs here:
      - Stress triggers (e.g. "gets anxious before mock tests")
      - What has helped before (e.g. "responds well to short daily goals")
      - Recurring patterns (e.g. "tends to skip Monday classes")
      - Stable personal facts (e.g. "works a part-time job in the evenings")

    Things that do NOT belong here (those go in generate_session_summary):
      - One-off events specific to this session
      - Decisions/commitments made just for this week
    """

    conversation_text = _student_text(messages)

    if not conversation_text:
        return ""

    prompt = f"""
From this coaching conversation, extract ONLY durable facts about the
student that would still be useful to know in a future, unrelated session.

Focus on:
- Stress triggers or sources of anxiety
- What has helped them before (study habits, coping strategies, etc.)
- Recurring patterns in behavior or performance
- Stable personal facts (schedule constraints, circumstances, preferences)

Do NOT include:
- One-off events that only matter for this session
- Decisions or commitments that are time-bound to this week

Output as a short bullet list. If nothing durable was shared, output exactly:
NONE

Conversation:

{conversation_text}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert at distilling durable facts about a "
                    "student from a coaching conversation, for use in future "
                    "sessions. Be conservative — only extract things that are "
                    "genuinely durable, not session-specific details."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    text = response.content.strip()
    if text.upper() == "NONE":
        return ""
    return text