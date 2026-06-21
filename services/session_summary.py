from langchain_core.messages import HumanMessage, SystemMessage

from llm.openai_client import get_llm

import json


llm = get_llm()

VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_URGENCIES = {"today", "tomorrow", "this_week"}


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


def extract_signal(messages: list[dict]) -> dict | None:
    """
    Analyze the session for a CONCERNING SIGNAL worth flagging to the coach —
    something that needs human attention, distinct from routine facts or the
    session narrative. Returns a structured dict or None if nothing concerning
    came up in this session.

    Returns (on a real signal):
        {
          "signal_type": str,   # short label, e.g. "Academic decline",
                                 # "Emotional distress", "Disengagement",
                                 # "Attendance risk", "Burnout risk", etc.
                                 # The LLM chooses the label that best fits —
                                 # no fixed enum, since concerns vary widely.
          "severity": "low" | "medium" | "high" | "critical",
          "urgency":  "today" | "tomorrow" | "this_week",
          "reason":   str,      # one or two plain-language sentences explaining
                                 # WHY this was flagged, specific to what the
                                 # student actually said/did this session.
        }

    Returns None if the session had nothing concerning enough to flag — most
    routine sessions should NOT produce a signal. This is intentionally a
    high bar, mirroring how extract_facts() is conservative about facts.
    """

    conversation_text = _student_text(messages)

    if not conversation_text:
        return None

    prompt = f"""
You are reviewing a coaching session transcript to decide whether the COACH
needs to be alerted about something concerning. Most sessions are routine
and should NOT produce a signal — only flag something if it genuinely
warrants a human coach's attention.

Look for things like:
- Signs of significant stress, anxiety, or emotional distress
- Academic decline that sounds serious or accelerating
- Disengagement, hopelessness, or motivation collapse
- Attendance or commitment risk (e.g. talking about dropping out, giving up)
- Anything safety-related or urgent in nature

severity guidance:
- low: worth knowing, not urgent
- medium: should be addressed in the next regular session
- high: coach should proactively reach out soon
- critical: coach should act today, this cannot wait

urgency guidance:
- today: coach should act today
- tomorrow: can wait until tomorrow
- this_week: can wait, but should happen this week

If NOTHING in this conversation rises to the level of needing coach
attention, respond with EXACTLY this and nothing else:
NONE

Otherwise, respond with ONLY a single valid JSON object (no markdown, no
explanation, no code fences) in exactly this shape:
{{"signal_type": "...", "severity": "low|medium|high|critical", "urgency": "today|tomorrow|this_week", "reason": "..."}}

Conversation:

{conversation_text}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert academic coaching risk-triage assistant. "
                    "You are conservative — you only flag genuine concerns, "
                    "never routine academic questions or normal conversation."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    text = response.content.strip()

    if text.upper() == "NONE":
        return None

    # Strip accidental code fences if the model adds them despite instructions
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        signal = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[extract_signal] Failed to parse LLM output as JSON: {exc!r} — raw: {text!r}")
        return None

    # Validate required fields are present and within expected vocab; if not,
    # fail safe (return None) rather than writing a malformed row to the sheet.
    required_fields = {"signal_type", "severity", "urgency", "reason"}
    if not required_fields.issubset(signal.keys()):
        print(f"[extract_signal] Missing fields in signal: {signal}")
        return None

    if signal["severity"] not in VALID_SEVERITIES:
        print(f"[extract_signal] Invalid severity {signal['severity']!r}, defaulting to 'medium'")
        signal["severity"] = "medium"

    if signal["urgency"] not in VALID_URGENCIES:
        print(f"[extract_signal] Invalid urgency {signal['urgency']!r}, defaulting to 'this_week'")
        signal["urgency"] = "this_week"

    return signal