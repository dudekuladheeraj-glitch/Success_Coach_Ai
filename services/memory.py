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

def _get_env(key: str) -> str:
    """Read from st.secrets first (Streamlit Cloud), then .env (local dev)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, "")

_memory_client = None


def get_memory_client():
    global _memory_client
    if _memory_client is not None:
        return _memory_client

    api_key = _get_env("MEM0_API_KEY")  # ← change from os.getenv

    if not api_key:
        raise ValueError("MEM0_API_KEY is not set in secrets or .env")

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


def _wait_until_indexed(
    student_id: str,
    mem_type: str,
    snippet: str,
    timeout_seconds: int = 30,
    poll_interval: float = 2.0,
) -> bool:
    """
    Mem0's add() is asynchronous on their backend — it returns
    {"status": "PENDING", "event_id": ...} immediately, before the memory
    is actually written/indexed. This polls get_all() until an entry with
    the right metadata type and matching text shows up, or times out.

    Returns True if confirmed indexed, False if it timed out (the memory
    may still land later — PENDING is not a failure, just "not yet").
    """
    import time

    if not snippet or not snippet.strip():
        return False

    # Use a short, distinctive slice of the text to match against, since
    # mem0 may rephrase/clean up the text during its own extraction.
    needle = snippet.strip()[:40].lower()

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            entries = _get_all(student_id)
        except Exception as exc:
            print(f"[memory] polling get_all failed: {exc}")
            entries = []

        for entry in entries:
            if _extract_metadata(entry).get("type") != mem_type:
                continue
            text = _extract_text(entry).lower()
            if needle[:20] in text or text[:20] in needle:
                return True

        time.sleep(poll_interval)

    return False


def save_session_memory(student_id: str, facts: str, summary: str) -> dict:
    """
    Convenience wrapper: save both memory types for one session in one call,
    then poll mem0 until each is actually confirmed indexed (or times out).

    Returns:
        {
          "fact_result":     <raw add() response, or None if no facts to save>,
          "summary_result":  <raw add() response>,
          "fact_confirmed":  bool,   # True once actually retrievable via get_all()
          "summary_confirmed": bool,
        }

    Use fact_confirmed / summary_confirmed in the UI instead of just checking
    that fact_result/summary_result is non-None — non-None only means
    "accepted by mem0", not "actually saved and queryable yet".
    """
    fact_result = save_factual_memory(student_id, facts)
    summary_result = save_session_summary(student_id, summary)

    fact_confirmed = (
        _wait_until_indexed(student_id, "fact", facts) if fact_result else False
    )
    summary_confirmed = _wait_until_indexed(student_id, "session_summary", summary)

    return {
        "fact_result": fact_result,
        "summary_result": summary_result,
        "fact_confirmed": fact_confirmed,
        "summary_confirmed": summary_confirmed,
    }


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


# ── Signals (M6) — stored in mem0 instead of the Sheet, since the sheet is
#    read-only and append_row() requires write/Editor access we don't have. ──


def save_signal(student_id: str, signal: dict):
    """
    Store a concerning-signal record from extract_signal(), tagged type='signal'.
    `signal` is the dict returned by extract_signal():
        {signal_type, severity, urgency, reason}

    actioned=False is set at write time, mirroring the sheet schema, so the
    alert panel / future M9 automation can filter to un-actioned signals and
    mark them actioned later via mark_signal_actioned().
    """
    if not signal:
        return None

    client = get_memory_client()
    # Store a readable text version as the memory content (what mem0 indexes
    # for search), and keep the structured fields in metadata (what our code
    # filters/reads from directly — text content may get rephrased by mem0's
    # own extraction, metadata won't).
    content = (
        f"Signal: {signal['signal_type']} (severity: {signal['severity']}, "
        f"urgency: {signal['urgency']}). Reason: {signal['reason']}"
    )
    print(f"[save_signal] Saving signal for {student_id}: {content}")
    messages = [{"role": "user", "content": content}]
    return client.add(
        messages,
        user_id=student_id,
        metadata={
            "type": "signal",
            "signal_type": signal["signal_type"],
            "severity": signal["severity"],
            "urgency": signal["urgency"],
            "reason": signal["reason"],
            "actioned": False,
        },
    )


def get_signals_for_student(student_id: str, include_actioned: bool = False) -> list[dict]:
    """
    All signal entries for one student, filtered to type='signal'.
    By default only un-actioned ones (mirrors get_signals() in sheets.py).

    Returns list of:
      {id, student_id, signal_type, severity, urgency, reason, actioned, created_at}
    """
    try:
        entries = _get_all(student_id)
    except Exception as exc:
        print(f"[memory] get_signals_for_student failed: {exc}")
        return []

    signals = []
    for entry in entries:
        meta = _extract_metadata(entry)
        if meta.get("type") != "signal":
            continue
        if not include_actioned and meta.get("actioned"):
            continue
        signals.append({
            "id": entry.get("id"),
            "student_id": student_id,
            "signal_type": meta.get("signal_type"),
            "severity": meta.get("severity"),
            "urgency": meta.get("urgency"),
            "reason": meta.get("reason"),
            "actioned": meta.get("actioned", False),
            "created_at": entry.get("created_at") or entry.get("updated_at"),
        })
    return signals


def get_all_active_signals_across_roster(student_ids: list[str]) -> list[dict]:
    """
    Roster-wide alert panel data. mem0 memories are scoped per user_id with
    no native cross-user query, so this loops get_signals_for_student() over
    every student in the roster and combines the un-actioned results.

    For a large roster this means one mem0 call per student — fine for
    typical cohort sizes, but if this ever gets slow, cache the result
    (the caller already does this via st.cache_data with a short TTL).
    """
    all_signals = []
    for sid in student_ids:
        try:
            all_signals.extend(get_signals_for_student(sid, include_actioned=False))
        except Exception as exc:
            print(f"[memory] failed to fetch signals for {sid}: {exc}")
            continue
    return all_signals


def mark_signal_actioned(student_id: str, signal_id: str) -> bool:
    """
    Mark one signal as actioned (e.g. coach clicked 'dismiss' / handled it).
    Uses mem0's update() to patch metadata. Returns True on success.

    NOTE: mem0's update() API shape varies by SDK version — if this fails,
    check `client.update.__doc__` / mem0's docs for your installed version
    and adjust the call below; the rest of the module doesn't depend on it.
    """
    if not signal_id:
        return False

    client = get_memory_client()
    try:
        client.update(memory_id=signal_id, metadata={"actioned": True})
        return True
    except Exception as exc:
        print(f"[memory] mark_signal_actioned failed: {exc}")
        return False