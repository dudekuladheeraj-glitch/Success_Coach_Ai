"""
services/daily_planner.py
──────────────────────────
M7: turn today's un-actioned signals across the roster into a structured
day for the coach — who needs attention, what kind of session, why, and
who gets deferred to tomorrow.

Pipeline:
  1. Pull roster (services.sheets.get_roster)
  2. Pull un-actioned signals for every student (services.memory)
  3. If nothing is flagged, return an empty plan (no LLM call needed)
  4. One LLM call: given the flagged students + their signals, the model
     decides how many slots make sense today, invents a session type per
     situation (no fixed list — see decision from planning conversation),
     assigns each flagged student a slot + reason, and defers anyone who
     doesn't fit with a reason why.
  5. Resolve abstract slot numbers into actual datetimes for calendar use.

This module does NOT touch the calendar — see services/calendar.py for
that. daily_planner only produces the structured plan; app.py wires the
two together.
"""

import json
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from llm.openai_client import get_llm
from services.memory import get_all_active_signals_across_roster
from services.sheets import get_roster

llm = get_llm()

VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_URGENCIES = {"today", "tomorrow", "this_week"}

# Coaching day window — used to convert the LLM's abstract slot numbers
# into real times. Kept simple and fixed for now; could become a config
# the coach sets later without changing the planning logic itself.
DAY_START_HOUR = 10   # 10:00 local
SLOT_DURATION_MINUTES = 30
SLOT_GAP_MINUTES = 0  # back-to-back; bump this if coaches want buffer time


def _get_roster_with_signals() -> tuple[list[dict], dict]:
    """
    Returns (roster, signals_by_student) where signals_by_student maps
    student_id -> list of un-actioned signal dicts for that student.
    """
    roster = get_roster()
    student_ids = [row["student_id"] for row in roster if row.get("student_id")]

    all_signals = get_all_active_signals_across_roster(student_ids)

    signals_by_student: dict[str, list[dict]] = {}
    for sig in all_signals:
        sid = sig.get("student_id")
        if not sid:
            continue
        signals_by_student.setdefault(sid, []).append(sig)

    return roster, signals_by_student


def _student_name_lookup(roster: list[dict]) -> dict:
    return {
        row["student_id"]: row.get("name", row["student_id"])
        for row in roster
        if row.get("student_id")
    }


def _severity_rank(sig: dict) -> int:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return order.get(sig.get("severity", "low"), 3)


def _build_flagged_summary(signals_by_student: dict, name_lookup: dict) -> list[dict]:
    """
    One entry per flagged student, carrying ALL their open signals (a
    student might have more than one) plus their highest-severity signal
    for sorting/display convenience.
    """
    flagged = []
    for sid, sigs in signals_by_student.items():
        sigs_sorted = sorted(sigs, key=_severity_rank)
        top = sigs_sorted[0]
        flagged.append({
            "student_id":    sid,
            "student_name":  name_lookup.get(sid, sid),
            "signals":       sigs_sorted,
            "top_severity":  top.get("severity", "low"),
            "top_urgency":   top.get("urgency", "this_week"),
        })
    flagged.sort(key=lambda f: _severity_rank({"severity": f["top_severity"]}))
    return flagged


def _build_prompt(flagged: list[dict], today: str) -> str:
    lines = [f"Today's date: {today}\n", "Flagged students (un-actioned signals):\n"]
    for f in flagged:
        lines.append(f"- student_id: {f['student_id']} (name: {f['student_name']})")
        for sig in f["signals"]:
            lines.append(
                f"    • {sig.get('signal_type', 'Signal')} "
                f"(severity: {sig.get('severity')}, urgency: {sig.get('urgency')}) "
                f"— {sig.get('reason', '')}"
            )
    flagged_block = "\n".join(lines)

    return f"""
You are building TODAY's coaching schedule for a student-success coach,
based on the flagged students and signals below.

{flagged_block}

Your job:
1. Decide how many coaching slots make sense for today, based on the
   number and severity of flagged students. Use your judgment — there is
   no fixed slot count. A normal day might be 4-8 slots; don't pad the
   plan with unnecessary slots if few students are flagged, and don't
   try to cram in everyone if many are critical (overflow should defer).
2. For each slot, assign exactly one flagged student. Prioritize by
   severity first (critical > high > medium > low), then urgency
   (today > tomorrow > this_week).
3. For each assigned student, invent a short, specific session_type label
   that fits THEIR situation (e.g. "Check-in", "Intervention", "Exam prep
   session", "Motivation reset", "Attendance conversation" — these are
   examples only, invent whatever best fits each case, do not reuse a
   fixed list mechanically).
4. Write a plain_reason — one or two sentences a coach can read in two
   seconds to understand WHY this student is on today's schedule, in
   plain language, not just restating the signal_type.
5. Any flagged student who does NOT get a slot today must be deferred:
   include them with a defer_reason explaining why they didn't fit
   (e.g. lower severity than others, already have multiple slots full
   ahead of them) — never silently drop a flagged student.
6. Order assigned_slots by priority (most urgent/severe first).

Respond with ONLY a single valid JSON object (no markdown, no code
fences, no explanation) in exactly this shape:

{{
  "assigned_slots": [
    {{
      "student_id": "...",
      "student_name": "...",
      "session_type": "...",
      "plain_reason": "...",
      "severity": "low|medium|high|critical",
      "urgency": "today|tomorrow|this_week"
    }}
  ],
  "deferred": [
    {{
      "student_id": "...",
      "student_name": "...",
      "defer_reason": "...",
      "severity": "low|medium|high|critical"
    }}
  ]
}}

If there are no flagged students at all, return:
{{"assigned_slots": [], "deferred": []}}
"""


def _parse_plan_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[daily_planner] Failed to parse LLM plan as JSON: {exc!r} — raw: {text!r}")
        return {"assigned_slots": [], "deferred": [], "parse_error": True}

    assigned = data.get("assigned_slots", [])
    deferred = data.get("deferred", [])

    # Defensive validation — fail safe per-item rather than discarding the
    # whole plan if the model produces one malformed entry.
    clean_assigned = []
    for item in assigned:
        if not isinstance(item, dict) or "student_id" not in item:
            print(f"[daily_planner] Dropping malformed assigned_slot: {item!r}")
            continue
        item.setdefault("student_name", item["student_id"])
        item.setdefault("session_type", "Check-in")
        item.setdefault("plain_reason", "")
        if item.get("severity") not in VALID_SEVERITIES:
            item["severity"] = "medium"
        if item.get("urgency") not in VALID_URGENCIES:
            item["urgency"] = "this_week"
        clean_assigned.append(item)

    clean_deferred = []
    for item in deferred:
        if not isinstance(item, dict) or "student_id" not in item:
            print(f"[daily_planner] Dropping malformed deferred item: {item!r}")
            continue
        item.setdefault("student_name", item["student_id"])
        item.setdefault("defer_reason", "Did not fit in today's available slots.")
        clean_deferred.append(item)

    return {"assigned_slots": clean_assigned, "deferred": clean_deferred, "parse_error": False}


def _resolve_slot_times(assigned_slots: list[dict], day: datetime = None) -> list[dict]:
    """
    Attach a real datetime + duration to each assigned slot, in order,
    back-to-back starting at DAY_START_HOUR. Pure function — does not
    touch the calendar, just prepares the field create_events_for_plan()
    in services/calendar.py expects.
    """
    day = day or datetime.now()
    start = day.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)

    resolved = []
    for i, slot in enumerate(assigned_slots):
        slot_time = start + timedelta(
            minutes=i * (SLOT_DURATION_MINUTES + SLOT_GAP_MINUTES)
        )
        resolved.append({
            **slot,
            "time": slot_time,
            "duration_minutes": SLOT_DURATION_MINUTES,
        })
    return resolved


def _is_urgent(signal: dict) -> bool:
    """
    Returns True if a signal is urgent enough to trigger an M9 plan update.
    Triggers on: critical (any urgency) OR high + urgency=today.
    """
    severity = signal.get("severity", "low")
    urgency  = signal.get("urgency", "this_week")
    if severity == "critical":
        return True
    if severity == "high" and urgency == "today":
        return True
    return False


def _slot_is_high_priority(slot: dict) -> bool:
    """
    Returns True if the student already in slot 1 is also critical/high+today,
    meaning we cannot auto-bump them without asking the coach.
    """
    severity = slot.get("severity", "low")
    urgency  = slot.get("urgency", "this_week")
    if severity == "critical":
        return True
    if severity == "high" and urgency == "today":
        return True
    return False


def update_plan_for_urgent_signal(
    plan: dict,
    student_id: str,
    student_name: str,
    signal: dict,
) -> dict:
    """
    M9: When a serious signal surfaces mid-day, update the already-generated
    daily plan to reflect the new urgency.

    Returns a result dict with one of three shapes:

    1. Signal not urgent enough — plan unchanged:
       {"plan_updated": False, "conflict": False, "plan": plan}

    2. Plan updated successfully — student inserted at slot 1:
       {"plan_updated": True, "conflict": False, "plan": <updated>,
        "change_summary": str}

    3. Conflict — slot 1 already has a critical/high+today student:
       {"plan_updated": False, "conflict": True,
        "existing_student": {id, name, severity, urgency, reason},
        "new_student":      {id, name, severity, urgency, reason},
        "plan": plan}
    """
    # ── Step 1: Is this signal urgent enough? ─────────────────────────────────
    if not _is_urgent(signal):
        return {"plan_updated": False, "conflict": False, "plan": plan}

    assigned = list(plan.get("assigned_slots", []))
    deferred = list(plan.get("deferred", []))

    new_slot_base = {
        "student_id":   student_id,
        "student_name": student_name,
        "session_type": "Urgent intervention",
        "plain_reason": signal.get("reason", "Serious concern flagged during session."),
        "severity":     signal.get("severity", "high"),
        "urgency":      signal.get("urgency", "today"),
        "duration_minutes": SLOT_DURATION_MINUTES,
    }

    # ── Step 2: Student already in assigned_slots? Promote to slot 1 ──────────
    existing_index = next(
        (i for i, s in enumerate(assigned) if s.get("student_id") == student_id),
        None,
    )
    if existing_index is not None:
        # Check conflict before promoting
        if existing_index != 0 and assigned and _slot_is_high_priority(assigned[0]):
            return {
                "plan_updated": False,
                "conflict": True,
                "existing_student": {
                    "id":       assigned[0]["student_id"],
                    "name":     assigned[0].get("student_name", assigned[0]["student_id"]),
                    "severity": assigned[0].get("severity", "unknown"),
                    "urgency":  assigned[0].get("urgency", "unknown"),
                    "reason":   assigned[0].get("plain_reason", ""),
                },
                "new_student": {
                    "id":       student_id,
                    "name":     student_name,
                    "severity": signal.get("severity"),
                    "urgency":  signal.get("urgency"),
                    "reason":   signal.get("reason", ""),
                },
                "plan": plan,
            }
        # Safe to promote — move to front
        student_slot = assigned.pop(existing_index)
        student_slot["session_type"] = "Urgent intervention"
        student_slot["plain_reason"] = signal.get("reason", student_slot["plain_reason"])
        student_slot["severity"]     = signal.get("severity", student_slot["severity"])
        student_slot["urgency"]      = signal.get("urgency", student_slot["urgency"])
        assigned.insert(0, student_slot)
        assigned = _resolve_slot_times(assigned)
        updated_plan = {**plan, "assigned_slots": assigned, "deferred": deferred}
        return {
            "plan_updated": True,
            "conflict": False,
            "plan": updated_plan,
            "change_summary": (
                f"{student_name} promoted to slot 1 (Urgent intervention) — "
                f"severity escalated to {signal.get('severity')}."
            ),
        }

    # ── Step 3: Student in deferred? Remove them (they're being escalated) ────
    deferred = [d for d in deferred if d.get("student_id") != student_id]

    # ── Step 4: Check slot 1 for conflict ─────────────────────────────────────
    if assigned and _slot_is_high_priority(assigned[0]):
        return {
            "plan_updated": False,
            "conflict": True,
            "existing_student": {
                "id":       assigned[0]["student_id"],
                "name":     assigned[0].get("student_name", assigned[0]["student_id"]),
                "severity": assigned[0].get("severity", "unknown"),
                "urgency":  assigned[0].get("urgency", "unknown"),
                "reason":   assigned[0].get("plain_reason", ""),
            },
            "new_student": {
                "id":       student_id,
                "name":     student_name,
                "severity": signal.get("severity"),
                "urgency":  signal.get("urgency"),
                "reason":   signal.get("reason", ""),
            },
            "plan": plan,
        }

    # ── Step 5: Safe to insert at slot 1 ──────────────────────────────────────
    # Record who was bumped (slot 1 occupant, if any) for the change summary
    bumped_name = None
    if assigned:
        bumped_name = assigned[0].get("student_name", assigned[0].get("student_id"))

    assigned.insert(0, new_slot_base)
    assigned = _resolve_slot_times(assigned)

    updated_plan = {
        **plan,
        "assigned_slots": assigned,
        "deferred": deferred,
    }

    if bumped_name:
        change_summary = (
            f"{student_name} added as slot 1 (Urgent intervention) — "
            f"{bumped_name} moved to slot 2."
        )
    else:
        change_summary = (
            f"{student_name} added as slot 1 (Urgent intervention) — "
            f"no existing slot 1 to displace."
        )

    return {
        "plan_updated": True,
        "conflict": False,
        "plan": updated_plan,
        "change_summary": change_summary,
    }


def resolve_conflict_pick(
    plan: dict,
    conflict: dict,
    prioritize: str,
) -> dict:
    """
    Called when the coach resolves a conflict by picking which student
    gets slot 1.

    prioritize: "existing" | "new"

    Returns:
        {"plan": <updated plan>, "change_summary": str}
    """
    assigned = list(plan.get("assigned_slots", []))
    deferred = list(plan.get("deferred", []))

    existing = conflict["existing_student"]
    new      = conflict["new_student"]

    new_slot_base = {
        "student_id":     new["id"],
        "student_name":   new["name"],
        "session_type":   "Urgent intervention",
        "plain_reason":   new["reason"],
        "severity":       new["severity"],
        "urgency":        new["urgency"],
        "duration_minutes": SLOT_DURATION_MINUTES,
    }

    if prioritize == "new":
        # Insert new student at slot 1; existing student stays at slot 2
        # (they were already slot 1, so they naturally shift down)
        assigned.insert(0, new_slot_base)
        assigned = _resolve_slot_times(assigned)
        change_summary = (
            f"{new['name']} placed at slot 1 (Urgent intervention). "
            f"{existing['name']} moved to slot 2 by coach decision."
        )
    else:
        # Keep existing at slot 1; append new student after all assigned slots
        # so they're seen today but not bumping the existing priority
        assigned.append(new_slot_base)
        assigned = _resolve_slot_times(assigned)
        change_summary = (
            f"{existing['name']} kept at slot 1 by coach decision. "
            f"{new['name']} added as last slot today."
        )

    updated_plan = {**plan, "assigned_slots": assigned, "deferred": deferred}
    return {"plan": updated_plan, "change_summary": change_summary}


def generate_daily_plan() -> dict:
    """
    Public entry point. Returns:
        {
          "assigned_slots": [
              {student_id, student_name, session_type, plain_reason,
               severity, urgency, time (datetime), duration_minutes}
          ],
          "deferred": [
              {student_id, student_name, defer_reason, severity}
          ],
          "flagged_count": int,   # total students with >=1 open signal
          "parse_error": bool,    # True if the LLM output couldn't be parsed
        }

    Returns an empty plan (no LLM call) if no students are currently
    flagged — most days with a healthy roster should hit this path.
    """
    roster, signals_by_student = _get_roster_with_signals()

    if not signals_by_student:
        return {"assigned_slots": [], "deferred": [], "flagged_count": 0, "parse_error": False}

    name_lookup = _student_name_lookup(roster)
    flagged = _build_flagged_summary(signals_by_student, name_lookup)

    today_str = datetime.now().strftime("%A, %Y-%m-%d")
    prompt = _build_prompt(flagged, today_str)

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are an expert academic-coaching scheduler. You turn a "
                "list of flagged students into a realistic, prioritized day "
                "for a single human coach. You are honest about capacity — "
                "you defer rather than overload the schedule."
            )
        ),
        HumanMessage(content=prompt),
    ])

    plan = _parse_plan_response(response.content)
    plan["assigned_slots"] = _resolve_slot_times(plan["assigned_slots"])
    plan["flagged_count"] = len(flagged)
    return plan