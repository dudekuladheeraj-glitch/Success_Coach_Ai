import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.openai_client import get_llm
from services.alerts import build_alerts
from services.sheets import (
    get_attendance,
    get_exam_schedules,
    get_exam_scores,
    get_signals,
    get_student_profile,
)

llm = get_llm()


# ── Node 1: Intent Classification ─────────────────────────────────────────────


def classify_intent(state: dict) -> dict:
    """
    Rule-based intent classifier.
    Buckets: exams | academics | overview | general
    """
    message = state["message"].lower()

    # "fetch/show/give me everything" style requests always get the full pull,
    # regardless of which other keywords are present.
    full_pull_phrases = [
        "all my data", "all data", "everything you have", "full details",
        "complete details", "all details", "fetch all", "show all", "give me all",
        "all my information", "all info",
    ]
    if any(phrase in message for phrase in full_pull_phrases):
        return {"intent": "overview"}

    if any(word in message for word in ["exam", "test", "schedule", "upcoming", "when is"]):
        intent = "exams"
    elif any(
        word in message
        for word in [
            "score", "marks", "grade", "attendance", "progress",
            "performance", "subject", "percentage", "how did i",
        ]
    ):
        intent = "academics"
    elif any(
        word in message
        for word in [
            "how am i", "overall", "summary", "status", "focus",
            "worry", "attention", "improve", "should i", "what should", "overview",
            "my details", "about me", "who am i", "my course",
            "what course", "my program", "what program", "profile", "tell me about",
            "manager", "coordinator", "contact",
        ]
    ):
        intent = "overview"
    else:
        # Even plain chitchat still gets the student's profile loaded —
        # see fetch_student_data, which always pulls at least the roster row.
        intent = "general"

    return {"intent": intent}


# ── Node 2: Fetch Student Data ─────────────────────────────────────────────────


def fetch_student_data(state: dict) -> dict:
    """
    Load ALL student data from Google Sheets on every turn.

    Earlier versions only fetched scores/attendance/exams/signals when the
    rule-based intent matched specific keywords. That silently broke for
    realistic phrasings the keyword lists didn't anticipate (e.g. "what's my
    cohort", "any alerts for me", "am I doing well") — those fell into
    "general" intent and got nothing but the bare profile, so the LLM had
    no choice but to say "I don't have that" even though the data existed.

    Fetching everything every turn is a handful of cheap Sheets reads and
    removes that entire class of bug. `intent` is still computed and passed
    through — it's used by build_alerts() to decide which alerts are most
    relevant to surface, not to gate what data gets loaded.
    """
    student_id = state["student_id"]

    profile = get_student_profile(student_id)
    context: dict = {
        "student_id": profile.get("student_id"),
        "name": profile.get("name"),
        "program": profile.get("program"),
        "cohort": profile.get("cohort"),
        "manager_email": profile.get("manager_email"),
    }

    # ── Always fetch everything ─────────────────────────────────────────────
    scores = get_exam_scores(student_id)
    attendance = get_attendance(student_id)  # list of weekly rows
    exams = get_exam_schedules(student_id)
    signals = get_signals(student_id)  # already filtered to un-actioned

    context["scores"] = scores
    context["attendance_history"] = attendance
    if attendance:
        context["latest_attendance_pct"] = attendance[-1].get("attendance_pct")
        context["latest_attendance_week"] = attendance[-1].get("week_of")
    context["exams"] = exams
    context["active_signals"] = signals

    # ── Build alerts from the full data set ─────────────────────────────────
    alerts = build_alerts(
        scores=scores,
        attendance=attendance,
        exams=exams,
        signals=signals,
    )

    return {
        "student_context": context,
        "alerts": alerts,
    }


# ── Node 3: Chat / LLM Response ───────────────────────────────────────────────


def chat_node(state: dict) -> dict:
    """
    Generate a coaching response using the student context, alerts, and
    prior conversation turns (so follow-ups like "explain in more detail"
    or "what about that" resolve correctly).
    """
    student_context = state.get("student_context", {})
    alerts = state.get("alerts", [])
    chat_history = state.get("chat_history", [])

    system_prompt = (
        "You are Success Coach AI — a warm, encouraging academic coach.\n\n"
        "Rules:\n"
        "- Answer naturally based on exactly what the student asked, using the "
        "conversation history below for context on follow-up questions.\n"
        "- Use ONLY the student data provided below for any facts about the "
        "student. This is the ONLY source of truth about the student — their "
        "name, program, cohort, scores, attendance, and exams. NEVER invent, "
        "assume, or guess any of these facts. If the data below doesn't include "
        "something the student asked about, say so plainly instead of making "
        "something up.\n"
        "- Exam scores are given as score/max_score pairs — compute percentages "
        "yourself when useful (e.g. 78/100 = 78%).\n"
        "- 'manager_email' in the student data is their assigned manager/coordinator's "
        "contact email — use it directly if asked who their manager is.\n"
        "- If 'student data' below is empty, tell the student you don't have their "
        "record loaded rather than answering as if you do.\n"
        "- For general knowledge questions unrelated to the student's record "
        "(e.g. 'what is AI'), answer normally — you are not limited to only "
        "discussing the student's data.\n"
        "- If the student asks about something unrelated to academics—such as movies, sports, entertainment, general trivia, celebrities, or games—politely decline to answer and redirect the conversation back to their studies. Do not provide any information or respond to the off-topic question itself, even briefly.\n"
        # "- For unrelated questions rather than the study content answer in such a way that you are not able to provide any information unrelated to the study content.\n"
        "- If alerts are listed, mention them proactively with encouragement "
        "and concrete next steps.\n"
        "- Keep responses concise and friendly 10-15 sentences unless more detail is needed.\n"
    )

    if student_context:
        system_prompt += (
            "\n\nStudent data for this conversation:\n"
            + json.dumps(student_context, indent=2, default=str)
        )

    if alerts:
        system_prompt += "\n\nItems that need attention:\n" + "\n".join(
            f"- {alert}" for alert in alerts
        )

    messages = [SystemMessage(content=system_prompt)]

    # Replay prior turns so the model has conversational context.
    for turn in chat_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["message"]))

    response = llm.invoke(messages)

    return {"response": response.content}