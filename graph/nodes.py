import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.openai_client import get_llm
from services.alerts import build_alerts
from services.knowledge_base import query_knowledge_base
from services.memory import get_factual_memory, get_session_count, get_session_summaries
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
    Buckets: exams | academics | overview | knowledge_base | briefing | general

    Order matters:
    1. Full data pull (highest priority)
    2. Briefing — "what did we discuss last time" style asks (checked early so
       it doesn't get swallowed by "overview"/"summary" keywords below)
    3. Academics  — checked BEFORE knowledge_base to avoid "attendance" being misrouted
    4. Exams
    5. Overview
    6. Knowledge base — platform questions (login, My Journey, Bonus Courses, etc.)
    7. General fallback
    """
    message = state["message"].lower()

    # ── 1. Full data pull ──────────────────────────────────────────────────────
    full_pull_phrases = [
        "all my data", "all data", "everything you have", "full details",
        "complete details", "all details", "fetch all", "show all", "give me all",
        "all my information", "all info",
    ]
    if any(phrase in message for phrase in full_pull_phrases):
        return {"intent": "overview"}

    # ── 2. Briefing — recap of PAST SESSIONS specifically ───────────────────────
    briefing_phrases = [
        "last session", "last time", "previous session", "previous sessions",
        "what did we discuss", "what did we talk about", "catch me up",
        "recap", "briefing", "what happened in our", "our last conversation",
        "earlier sessions", "past sessions", "session history",
    ]
    if any(phrase in message for phrase in briefing_phrases):
        return {"intent": "briefing"}

    # ── 3. Academics (checked BEFORE knowledge_base) ───────────────────────────
    if any(word in message for word in [
        "score", "marks", "grade", "attendance", "progress",
        "performance", "subject", "percentage", "how did i",
    ]):
        return {"intent": "academics"}

    # ── 4. Exams ───────────────────────────────────────────────────────────────
    if any(word in message for word in [
        "exam", "test", "upcoming", "when is",
    ]):
        return {"intent": "exams"}

    # ── 5. Overview ────────────────────────────────────────────────────────────
    if any(word in message for word in [
        "how am i", "overall", "summary", "status", "focus",
        "worry", "attention", "improve", "should i", "what should",
        "overview", "my details", "about me", "who am i", "my course",
        "what course", "my program", "what program", "profile",
        "tell me about", "manager", "coordinator", "contact",
    ]):
        return {"intent": "overview"}

    # ── 6. Knowledge base — platform questions only ────────────────────────────
    kb_phrases = [
        "login", "log in", "sign in", "learning portal",
        "my journey", "growth cycle", "bonus course", "bookmark",
        "lastminute", "last minute", "search option", "home page",
        "otp", "ccbp", "induction", "induction video",
        "nxtwave", "nxtmock", "topin", "course library",
        "how do i", "how to", "where do i", "where can i",
        "how can i", "what is my journey", "what are bonus",
        "placement", "off campus", "mock interview", "leaderboard",
        "navigation", "portal url", "mobile number", "registered",
    ]
    if any(phrase in message for phrase in kb_phrases):
        return {"intent": "knowledge_base"}

    # ── 7. General fallback ────────────────────────────────────────────────────
    return {"intent": "general"}


# ── Node 2: Fetch Student Data ─────────────────────────────────────────────────


def fetch_student_data(state: dict) -> dict:
    """
    Load ALL student data from Google Sheets on every turn, PLUS the
    student's long-term memory from mem0 (M5):

      - known_facts:      durable facts (stress triggers, what's helped,
                           recurring patterns), pulled on every turn so they
                           always shape how the AI talks to the student.
      - session_summaries: past session recaps — fetched here too, but only
                           actually surfaced to the LLM by the dedicated
                           briefing_node when intent == "briefing", to avoid
                           bloating every prompt with old session detail.
      - session_number:   how many sessions came before this one, so the AI
                           can tell a 1st-session student apart from a 5th.

    intent is still passed through for build_alerts() and routing.
    """
    student_id = state["student_id"]
    message = state.get("message", "")

    profile = get_student_profile(student_id)
    context: dict = {
        "student_id": profile.get("student_id"),
        "name":          profile.get("name"),
        "program":       profile.get("program"),
        "cohort":        profile.get("cohort"),
        "manager_email": profile.get("manager_email"),
    }

    scores     = get_exam_scores(student_id)
    attendance = get_attendance(student_id)
    exams      = get_exam_schedules(student_id)
    signals    = get_signals(student_id)

    context["scores"]             = scores
    context["attendance_history"] = attendance
    if attendance:
        context["latest_attendance_pct"]  = attendance[-1].get("attendance_pct")
        context["latest_attendance_week"] = attendance[-1].get("week_of")
    context["exams"]          = exams
    context["active_signals"] = signals

    alerts = build_alerts(
        scores=scores,
        attendance=attendance,
        exams=exams,
        signals=signals,
    )

    # ── Long-term memory (M5) ───────────────────────────────────────────────
    try:
        known_facts = get_factual_memory(student_id, query=message)
    except Exception as exc:
        print(f"[fetch_student_data] get_factual_memory failed: {exc}")
        known_facts = []

    try:
        past_sessions = get_session_count(student_id)
    except Exception as exc:
        print(f"[fetch_student_data] get_session_count failed: {exc}")
        past_sessions = 0

    context["known_facts"]    = known_facts
    context["session_number"] = past_sessions + 1  # this session

    return {
        "student_context": context,
        "alerts":          alerts,
        "known_facts":     known_facts,
    }


# ── Node 3: Knowledge Base Query ──────────────────────────────────────────────


def knowledge_base_node(state: dict) -> dict:
    """
    Query ChromaDB for chunks relevant to the student's question,
    then generate a grounded answer using only those chunks.

    Falls back gracefully to the normal chat_node behaviour if
    ChromaDB is not available or returns no results.
    """
    message = state["message"]

    kb_context = query_knowledge_base(message)

    if not kb_context:
        # No results — fall back to the regular chat node
        return chat_node(state)

    student_context = state.get("student_context", {})
    chat_history    = state.get("chat_history", [])

    system_prompt = (
        "You are Success Coach AI — a warm, encouraging academic coach.\n\n"
        "The student has asked a question about the NxtWave learning platform. "
        "Answer using ONLY the knowledge base excerpts provided below. "
        "Do not invent features or steps that are not mentioned in the excerpts. "
        "If the excerpts don't contain enough information to fully answer, say so "
        "and suggest the student contact their coordinator.\n\n"
        "Keep answers clear, friendly, and step-by-step where relevant.\n"
    )

    if student_context.get("name"):
        system_prompt += f"\nStudent's name: {student_context['name']}\n"

    system_prompt += (
        "\n\nRelevant knowledge base excerpts:\n"
        "────────────────────────────────────\n"
        + kb_context
        + "\n────────────────────────────────────\n"
    )

    messages = [SystemMessage(content=system_prompt)]

    for turn in chat_history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return {"response": response.content}


# ── Node 4: Briefing (past session recap) ─────────────────────────────────────


def briefing_node(state: dict) -> dict:
    """
    Answer "what did we discuss last time" / "catch me up" style requests
    using stored SESSION SUMMARIES (not factual memory, not live sheet data).

    Falls back to chat_node if no past sessions exist yet.
    """
    student_id = state["student_id"]
    message    = state["message"]

    try:
        summaries = get_session_summaries(student_id, limit=5)
    except Exception as exc:
        print(f"[briefing_node] get_session_summaries failed: {exc}")
        summaries = []

    if not summaries:
        return chat_node(state)

    student_context = state.get("student_context", {})
    chat_history    = state.get("chat_history", [])

    system_prompt = (
        "You are Success Coach AI — a warm, encouraging academic coach.\n\n"
        "The student is asking for a recap of PAST SESSIONS. Use ONLY the "
        "session summaries below to answer. Summarize them clearly, most "
        "recent first, in plain conversational language. Do not invent "
        "details that aren't in the summaries below.\n"
    )

    if student_context.get("name"):
        system_prompt += f"\nStudent's name: {student_context['name']}\n"

    system_prompt += (
        "\n\nPast session summaries (most recent first):\n"
        "────────────────────────────────────\n"
        + "\n\n".join(f"Session: {s}" for s in summaries)
        + "\n────────────────────────────────────\n"
    )

    messages = [SystemMessage(content=system_prompt)]

    for turn in chat_history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return {"response": response.content}


# ── Node 5: Chat / LLM Response ───────────────────────────────────────────────


def chat_node(state: dict) -> dict:
    """
    Generate a coaching response using student context, alerts,
    long-term memory (known facts + session count), and prior
    conversation turns.
    """
    student_context = state.get("student_context", {})
    alerts          = state.get("alerts", [])
    chat_history     = state.get("chat_history", [])
    known_facts      = state.get("known_facts") or student_context.get("known_facts", [])
    session_number   = student_context.get("session_number", 1)

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
        "(e.g. 'what is AI'), answer normally.\n"
        "- If the student asks about something unrelated to academics — such as "
        "movies, sports, entertainment, general trivia, celebrities, or games — "
        "politely decline and redirect to their studies.\n"
        "- If alerts are listed, mention them proactively with encouragement "
        "and concrete next steps.\n"
        "- If 'Known facts about this student' are listed below, let them quietly "
        "shape your tone and advice (e.g. avoid a known stress trigger, suggest a "
        "strategy that's worked before) — don't just recite them back verbatim.\n"
        "- Keep responses concise and friendly (10–15 sentences unless more detail "
        "is needed).\n"
    )

    if student_context:
        system_prompt += (
            "\n\nStudent data for this conversation:\n"
            + json.dumps(student_context, indent=2, default=str)
        )

    if known_facts:
        system_prompt += "\n\nKnown facts about this student from past sessions:\n" + "\n".join(
            f"- {fact}" for fact in known_facts
        )

    if session_number and session_number > 1:
        system_prompt += (
            f"\n\nThis is session #{session_number} with this student. "
            "Adjust your tone accordingly — more rapport, less re-explaining "
            "of basics, and reference relevant history naturally where it helps."
        )

    if alerts:
        system_prompt += "\n\nItems that need attention:\n" + "\n".join(
            f"- {alert}" for alert in alerts
        )

    messages = [SystemMessage(content=system_prompt)]

    for turn in chat_history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["message"]))

    response = llm.invoke(messages)
    return {"response": response.content}