
"""
services/alerts.py
───────────────────
Builds actionable alert strings from student data, matching the
confirmed sheet schema (score/max_score pairs, weekly attendance,
signal_sheet with severity/urgency).
"""

from datetime import datetime
from collections import defaultdict


def _parse_date(value):
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _score_pct(score, max_score) -> float | None:
    try:
        s, m = float(score), float(max_score)
        if m == 0:
            return None
        return round((s / m) * 100, 1)
    except (ValueError, TypeError):
        return None


def build_score_alerts(scores: list[dict]) -> list[str]:
    """
    scores: list of {subject, score, max_score, date} sorted oldest → newest
    (as returned by services.sheets.get_exam_scores).

    Compares the latest two attempts per subject; flags a drop > 5 percentage points.
    """
    alerts = []
    by_subject = defaultdict(list)
    for row in scores:
        by_subject[row.get("subject", "Unknown")].append(row)

    for subject, rows in by_subject.items():
        if len(rows) < 2:
            continue
        # rows already sorted oldest -> newest globally; take last two for this subject
        previous, current = rows[-2], rows[-1]
        prev_pct = _score_pct(previous.get("score"), previous.get("max_score"))
        curr_pct = _score_pct(current.get("score"), current.get("max_score"))
        if prev_pct is None or curr_pct is None:
            continue
        if curr_pct < prev_pct - 5:
            alerts.append(
                f"📉 {subject} score dropped: {prev_pct}% → {curr_pct}%"
            )
    return alerts


def build_attendance_alerts(attendance_records: list[dict]) -> list[str]:
    """
    attendance_records: list of {week_of, classes_scheduled, classes_attended, attendance_pct}
    sorted oldest → newest (as returned by services.sheets.get_attendance).

    Flags if the most recent week's attendance_pct < 75.
    """
    alerts = []
    if not attendance_records:
        return alerts
    latest = attendance_records[-1]
    try:
        pct = float(latest.get("attendance_pct"))
        if pct < 75:
            week = latest.get("week_of", "this week")
            alerts.append(f"⚠️ Low attendance for week of {week}: {pct}%")
    except (ValueError, TypeError):
        pass
    return alerts


def build_exam_alerts(exams: list[dict]) -> list[str]:
    """
    exams: list of {subject, exam_date, exam_type} (as returned by
    services.sheets.get_exam_schedules).

    Flags exams within the next 7 days.
    """
    alerts = []
    today = datetime.now().date()
    for exam in exams:
        subject = exam.get("subject", "Subject")
        exam_type = exam.get("exam_type", "")
        exam_date = _parse_date(exam.get("exam_date"))
        if not exam_date:
            continue
        days_left = (exam_date - today).days
        if 0 <= days_left <= 7:
            type_label = f" ({exam_type})" if exam_type else ""
            alerts.append(
                f"📅 {subject}{type_label} exam in {days_left} day(s) — {exam_date.isoformat()}"
            )
    return alerts


def build_signal_alerts(signals: list[dict]) -> list[str]:
    """
    signals: list of {signal_type, severity, urgency, reason, timestamp, actioned}
    — already filtered to un-actioned signals by services.sheets.get_signals.
    """
    alerts = []
    for sig in signals:
        signal_type = sig.get("signal_type", "Signal")
        severity = sig.get("severity", "")
        reason = sig.get("reason", "")
        label = f"🚩 {signal_type}"
        if severity:
            label += f" ({severity})"
        if reason:
            label += f": {reason}"
        alerts.append(label)
    return alerts


def build_alerts(
    scores: list[dict] = None,
    attendance: list[dict] = None,
    exams: list[dict] = None,
    signals: list[dict] = None,
) -> list[str]:
    """
    Combine all alert types. Any argument can be omitted/empty depending
    on which data was fetched for the current intent.
    """
    alerts = []
    alerts += build_attendance_alerts(attendance or [])
    alerts += build_score_alerts(scores or [])
    alerts += build_exam_alerts(exams or [])
    alerts += build_signal_alerts(signals or [])
    return alerts