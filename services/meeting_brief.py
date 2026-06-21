"""
services/meeting_brief.py
───────────────────────────
M8: pre-meeting brief on any student, for the coach.

Pulls together everything built in M5 (factual memory + session summaries)
and M6 (signal history) plus live sheet data, and asks the LLM to produce a
focused 4-section brief a coach can read in under a minute before a
1-on-1: current situation, what's changed since the last session, open
concerns, and conversation starters.

This is coach-facing only — never shown to the student, and intentionally
separate from chat_node's system prompt (which is conversational, not a
prep document).
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from llm.openai_client import get_llm
from services.memory import (
    get_factual_memory,
    get_session_summaries,
    get_signals_for_student,
)
from services.sheets import (
    get_attendance,
    get_exam_schedules,
    get_exam_scores,
    get_student_profile,
)

llm = get_llm()


def _gather_context(student_id: str) -> dict:
    """
    Collect every data source the brief draws from. Each call is wrapped
    individually so one missing/erroring source (e.g. student not in sheet
    yet, or mem0 hiccup) doesn't take down the whole brief — it just shows
    up as an empty section instead of crashing the page.
    """
    context: dict = {"student_id": student_id}

    try:
        context["profile"] = get_student_profile(student_id)
    except Exception as exc:
        print(f"[meeting_brief] get_student_profile failed: {exc}")
        context["profile"] = {}

    try:
        context["scores"] = get_exam_scores(student_id)
    except Exception as exc:
        print(f"[meeting_brief] get_exam_scores failed: {exc}")
        context["scores"] = []

    try:
        context["attendance"] = get_attendance(student_id)
    except Exception as exc:
        print(f"[meeting_brief] get_attendance failed: {exc}")
        context["attendance"] = []

    try:
        context["exams"] = get_exam_schedules(student_id)
    except Exception as exc:
        print(f"[meeting_brief] get_exam_schedules failed: {exc}")
        context["exams"] = []

    try:
        context["known_facts"] = get_factual_memory(student_id)
    except Exception as exc:
        print(f"[meeting_brief] get_factual_memory failed: {exc}")
        context["known_facts"] = []

    try:
        # Most recent few sessions, not the entire history — keeps the
        # prompt focused on "what's changed lately" rather than everything
        # that's ever happened.
        context["recent_sessions"] = get_session_summaries(student_id, limit=3)
    except Exception as exc:
        print(f"[meeting_brief] get_session_summaries failed: {exc}")
        context["recent_sessions"] = []

    try:
        # include_actioned=True on purpose — even a RESOLVED signal from last
        # week is useful "what's changed" context for the coach, not just
        # currently-open ones.
        context["signal_history"] = get_signals_for_student(
            student_id, include_actioned=True
        )
    except Exception as exc:
        print(f"[meeting_brief] get_signals_for_student failed: {exc}")
        context["signal_history"] = []

    return context


def _build_prompt(context: dict) -> str:
    profile = context["profile"]
    name = profile.get("name", context["student_id"])

    scores_block = "\n".join(
        f"  - {s.get('subject')}: {s.get('score')}/{s.get('max_score')} on {s.get('date')}"
        for s in context["scores"]
    ) or "  (no exam scores on record)"

    attendance_block = "\n".join(
        f"  - week of {a.get('week_of')}: {a.get('attendance_pct')}%"
        for a in context["attendance"][-4:]  # last few weeks is plenty context
    ) or "  (no attendance on record)"

    exams_block = "\n".join(
        f"  - {e.get('subject')} ({e.get('exam_type')}) on {e.get('exam_date')}"
        for e in context["exams"]
    ) or "  (no upcoming exams on record)"

    facts_block = "\n".join(f"  - {f}" for f in context["known_facts"]) or "  (none recorded yet)"

    sessions_block = "\n\n".join(
        f"  Session: {s}" for s in context["recent_sessions"]
    ) or "  (no prior sessions recorded — this would be a first session)"

    signals_block = "\n".join(
        f"  - [{'ACTIONED' if sig.get('actioned') else 'OPEN'}] "
        f"{sig.get('signal_type')} (severity: {sig.get('severity')}, "
        f"urgency: {sig.get('urgency')}) — {sig.get('reason')}"
        for sig in context["signal_history"]
    ) or "  (no signals on record)"

    return f"""
You are preparing a coach for a 1-on-1 meeting with a student. Produce a
focused brief the coach can read in under a minute, right before walking
into the room.

Student: {name} ({context['student_id']})

── Exam scores ──
{scores_block}

── Recent attendance ──
{attendance_block}

── Upcoming exams ──
{exams_block}

── Known facts about this student (from past sessions) ──
{facts_block}

── Recent session summaries (most recent first) ──
{sessions_block}

── Signal history (concerns flagged in past sessions) ──
{signals_block}

Using ONLY the information above, produce a brief with exactly these four
sections, using these exact headers:

## Current Situation
A short, factual snapshot of where this student stands academically right
now (scores, attendance, upcoming exams) — 2-4 sentences.

## What's Changed Since Last Session
Compare the current data above against what the most recent session
summary said. Call out anything new, improved, or worsened. If there are
no prior sessions, say this is the first session and skip comparison.

## Open Concerns
Concerns that are still relevant — draw from known facts AND open
(un-actioned) signals. Do not repeat resolved/actioned signals as if they
are still active concerns, but you may mention them briefly as recent
history if relevant context.

## Conversation Starters
2-3 specific, natural opening lines or questions the coach could use to
start the conversation, grounded in the actual data above — not generic
small talk. Reference something concrete (a score, a fact, a prior
commitment) wherever possible.

Keep the whole brief concise — a coach should be able to read it in under
a minute. Do not invent information not present above. If a section has
nothing relevant to say, say so briefly rather than padding it.
"""


def generate_brief(student_id: str) -> dict:
    """
    Public entry point. Returns:
        {
          "student_id": str,
          "student_name": str,
          "brief_text": str,      # the LLM's full markdown brief
          "has_history": bool,    # False if this would be a first session
        }
    """
    context = _gather_context(student_id)
    prompt = _build_prompt(context)

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are an expert academic-coaching assistant preparing a "
                "concise, accurate pre-meeting brief for a human coach. You "
                "never invent facts not present in the provided data."
            )
        ),
        HumanMessage(content=prompt),
    ])

    return {
        "student_id": student_id,
        "student_name": context["profile"].get("name", student_id),
        "brief_text": response.content,
        "has_history": bool(context["recent_sessions"]),
    }