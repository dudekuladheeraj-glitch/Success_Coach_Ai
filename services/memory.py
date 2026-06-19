"""
services/memory.py
───────────────────
Mem0-backed long-term memory for Success Coach AI.

Two memory types are stored in the SAME mem0 user bucket (user_id=student_id)
but tagged with different metadata so they can be retrieved separately:

  type="fact"             → durable facts about the student (stress triggers,
                             what has helped, recurring patterns). Used on
                             EVERY turn to shape how the AI talks to the student.

  type="session_summary"  → what was discussed/decided in a specific session.
                             Used only when the coach/student explicitly asks
                             for a recap/briefing of past sessions.

NOTE: mem0's SDK has changed shape across versions. The helpers below try the
modern `client.search(...)` / `client.get_all(...)` signatures first and fall
back to older ones. If your installed `mem0ai` version differs, adjust the
`_search` / `_get_all` calls — the rest of this module doesn't need to change.
"""

import os

from mem0 import MemoryClient

_memory_client = None


def get_memory_client():
    global _memory_client

    if _memory_client is not None:
        return _memory_client

    api_key = os.getenv("MEM0_API_KEY")

    if not api_key:
        raise ValueError("MEM0_API_KEY is not set.")

    _memory_client = MemoryClient(api_key=api_key)
    return _memory_client


# ── Internal helpers (version-tolerant wrappers around mem0) ──────────────────


def _get_all(student_id: str) -> list[dict]:
    """Return every raw memory entry mem0 has for this student."""
    client = get_memory_client()
    try:
        results = client.get_all(user_id=student_id)
    except TypeError:
        # Some versions expect filters={"user_id": ...} instead
        results = client.get_all(filters={"user_id": student_id})

    # mem0 sometimes wraps results as {"results": [...]}
    if isinstance(results, dict):
        results = results.get("results", [])
    return results or []


def _search(student_id: str, query: str, limit: int = 10) -> list[dict]:
    """Semantic search over this student's memories."""
    client = get_memory_client()
    try:
        results = client.search(query=query, user_id=student_id, limit=limit)
    except TypeError:
        results = client.search(query=query, user_id=student_id)

    if isinstance(results, dict):
        results = results.get("results", [])
    return results or []


def _extract_text(entry: dict) -> str:
    """mem0 entries usually expose the text under 'memory', sometimes 'text'."""
    return entry.get("memory") or entry.get("text") or ""


def _extract_metadata(entry: dict) -> dict:
    return entry.get("metadata") or {}


# ── Write paths ────────────────────────────────────────────────────────────────


def save_factual_memory(student_id: str, facts: str):
    """
    Store durable facts about the student (stress triggers, what has helped,
    recurring patterns, important facts they've shared). Tagged type='fact'.
    """
    if not facts or not facts.strip():
        return None

    client = get_memory_client()
    messages = [{"role": "user", "content": facts}]

    return client.add(
        messages,
        user_id=student_id,
        metadata={"type": "fact"},
    )


def save_session_summary(student_id: str, summary: str):
    """
    Store a narrative summary of one coaching session (what was discussed,
    decided, committed to). Tagged type='session_summary'.
    """
    if not summary or not summary.strip():
        return None

    client = get_memory_client()
    messages = [{"role": "user", "content": summary}]

    return client.add(
        messages,
        user_id=student_id,
        metadata={"type": "session_summary"},
    )


def save_session_memory(student_id: str, facts: str, summary: str) -> dict:
    """
    Convenience wrapper: save both memory types for one session in one call.
    Use this from the 'End Session' button instead of calling the two
    functions above separately.
    """
    fact_result = save_factual_memory(student_id, facts)
    summary_result = save_session_summary(student_id, summary)
    return {"fact_result": fact_result, "summary_result": summary_result}


# ── Read paths ─────────────────────────────────────────────────────────────────


def get_factual_memory(student_id: str, query: str = "", limit: int = 8) -> list[str]:
    """
    Return known facts about the student, most relevant to `query` if given,
    otherwise just the most recent facts. Filters to type='fact' only.
    """
    try:
        if query and query.strip():
            entries = _search(student_id, query, limit=limit * 2)
        else:
            entries = _get_all(student_id)
    except Exception as exc:
        print(f"[memory] get_factual_memory failed: {exc}")
        return []

    facts = [
        _extract_text(e)
        for e in entries
        if _extract_metadata(e).get("type") == "fact" and _extract_text(e)
    ]
    return facts[:limit]


def get_session_summaries(student_id: str, limit: int = 10) -> list[str]:
    """
    Return past session summaries (most recent first if mem0 preserves
    insertion order; otherwise unordered). Filters to type='session_summary'.
    """
    try:
        entries = _get_all(student_id)
    except Exception as exc:
        print(f"[memory] get_session_summaries failed: {exc}")
        return []

    summaries = [
        _extract_text(e)
        for e in entries
        if _extract_metadata(e).get("type") == "session_summary" and _extract_text(e)
    ]
    return summaries[-limit:][::-1]  # most recent last-in -> first-out


def get_session_count(student_id: str) -> int:
    """
    Number of completed sessions for this student, based on how many
    session_summary entries mem0 has stored.
    """
    try:
        return len(get_session_summaries(student_id, limit=10_000))
    except Exception as exc:
        print(f"[memory] get_session_count failed: {exc}")
        return 0