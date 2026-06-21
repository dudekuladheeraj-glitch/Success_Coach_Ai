"""
services/sheets.py
──────────────────
Google Sheets data layer for Success Coach AI.

CONFIRMED SCHEMA (verified directly against the spreadsheet):

  roster
    student_id | name | program | cohort | manager_email

  exam_scores
    student_id | subject | score | max_score | date

  attendance   (ONE ROW PER WEEK per student — not a single row!)
    student_id | week_of | classes_scheduled | classes_attended | attendance_pct

  exam_schedule
    student_id | subject | exam_date | exam_type

  signal_sheet
    student_id | signal_type | severity | urgency | reason | timestamp | actioned

Auth priority:
  1. Service account  (credentials.json with "type": "service_account")
  2. OAuth flow       (credentials.json with "installed" key  →  saves token.json)
"""

import json
import os
from datetime import datetime

import gspread
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID")

# ── Module-level cache (plain globals — avoids stale-token lru_cache bugs) ────
_client: gspread.Client | None = None
_spreadsheet = None


def _get_client() -> gspread.Client:
    global _client
    if _client is not None:
        return _client

    # ── Streamlit Cloud: read from st.secrets ──────────────────────────────
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            from google.oauth2.service_account import Credentials
            creds = Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),  # convert AttrDict → plain dict
                scopes=SCOPES,
            )
            _client = gspread.authorize(creds)
            return _client

    except Exception:
        pass  # not running in Streamlit, fall through to file-based auth

    # ── Local dev: read from credentials.json ──────────────────────────────
    creds_path = "credentials.json"
    if not os.path.exists(creds_path):
        raise FileNotFoundError(
            "credentials.json not found. Place it in the project root.\n"
            "Get it from Google Cloud Console → APIs & Services → Credentials."
        )

    with open(creds_path) as f:
        creds_data = json.load(f)

    cred_type = creds_data.get("type", "")

    if cred_type == "service_account":
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(creds_data, scopes=SCOPES)
        _client = gspread.authorize(creds)
    else:
        token_path = "token.json"
        if os.path.exists(token_path):
            try:
                with open(token_path) as f:
                    tok = json.load(f)
                required = {"client_id", "client_secret", "refresh_token"}
                if not required.issubset(tok.keys()):
                    os.remove(token_path)
            except Exception:
                os.remove(token_path)

        _client = gspread.oauth(
            scopes=SCOPES,
            credentials_filename=creds_path,
            authorized_user_filename=token_path,
        )

    return _client


def _get_spreadsheet():
    global _spreadsheet
    if _spreadsheet is not None:
        return _spreadsheet

    # Try st.secrets first (Streamlit Cloud), then fall back to .env
    sheet_id = None
    try:
        import streamlit as st
        sheet_id = st.secrets.get("GOOGLE_SPREADSHEET_ID")
    except Exception:
        pass
    sheet_id = sheet_id or SPREADSHEET_ID  # SPREADSHEET_ID comes from os.getenv()

    if not sheet_id:
        raise ValueError(
            "GOOGLE_SPREADSHEET_ID is not set.\n"
            "Set it in .streamlit/secrets.toml (local) or Streamlit Cloud secrets."
        )
    _spreadsheet = _get_client().open_by_key(sheet_id)
    return _spreadsheet


# Opening the tabs in the spreadsheet
def _worksheet(tab_name: str):
    """Return a worksheet, with a clear error listing available tabs if the name is wrong."""
    try:
        return _get_spreadsheet().worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        available = [ws.title for ws in _get_spreadsheet().worksheets()]
        raise ValueError(
            f"Tab '{tab_name}' not found in spreadsheet.\n"
            f"Available tabs: {available}"
        )


# Getting the rows by student_id from the each tab
def _rows_by_student(tab_name: str, student_id: str) -> list[dict]:
    """All rows where student_id matches (case-insensitive)."""
    rows = _worksheet(tab_name).get_all_records()
    sid = student_id.strip().upper()
    return [
        row for row in rows
        if str(row.get("student_id", "")).strip().upper() == sid
    ]


# Parsing the date from the rows
def _parse_date(value):
    """Parse a date string into a date object, trying common formats."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# ── Public API ────────────────────────────────────────────────────────────────


def get_roster() -> list[dict]:
    """All students from the 'roster' tab."""
    return _worksheet("roster").get_all_records()


def get_student_profile(student_id: str) -> dict:
    """Single student profile. Raises ValueError if not found."""
    rows = _rows_by_student("roster", student_id)
    if not rows:
        raise ValueError(
            f"Student '{student_id}' not found in roster tab.\n"
            f"Check the student_id column in your sheet."
        )
    return rows[0]


def get_exam_scores(student_id: str) -> list[dict]:
    """
    Scores from 'exam_scores' tab.
    Columns: student_id | subject | score | max_score | date
    Returns scores sorted by date (oldest → newest) so consecutive
    rows for the same subject represent previous → current.
    """
    rows = _rows_by_student("exam_scores", student_id)
    parsed = []
    for row in rows:
        parsed.append({
            "subject": row.get("subject"),
            "score": row.get("score"),
            "max_score": row.get("max_score"),
            "date": row.get("date"),
            "_date_obj": _parse_date(row.get("date")),
        })
    parsed.sort(key=lambda r: (r["_date_obj"] is None, r["_date_obj"]))
    for r in parsed:
        r.pop("_date_obj", None)
    return parsed


def get_attendance(student_id: str) -> list[dict]:
    """
    Attendance from 'attendance' tab — ONE ROW PER WEEK.
    Columns: student_id | week_of | classes_scheduled | classes_attended | attendance_pct
    Returns all weekly rows, sorted by week_of (oldest → newest).
    """
    rows = _rows_by_student("attendance", student_id)
    parsed = []
    for row in rows:
        parsed.append({
            "week_of": row.get("week_of"),
            "classes_scheduled": row.get("classes_scheduled"),
            "classes_attended": row.get("classes_attended"),
            "attendance_pct": row.get("attendance_pct"),
            "_date_obj": _parse_date(row.get("week_of")),
        })
    parsed.sort(key=lambda r: (r["_date_obj"] is None, r["_date_obj"]))
    for r in parsed:
        r.pop("_date_obj", None)
    return parsed


def get_latest_attendance_pct(student_id: str) -> float | None:
    """Convenience helper: attendance_pct from the most recent week_of row."""
    records = get_attendance(student_id)
    if not records:
        return None
    latest = records[-1]
    try:
        return float(latest["attendance_pct"])
    except (ValueError, TypeError):
        return None


def get_exam_schedules(student_id: str) -> list[dict]:
    """
    Exams from 'exam_schedule' tab.
    Columns: student_id | subject | exam_date | exam_type
    Returns rows sorted by exam_date (soonest first).
    """
    rows = _rows_by_student("exam_schedule", student_id)
    parsed = []
    for row in rows:
        parsed.append({
            "subject": row.get("subject"),
            "exam_date": row.get("exam_date"),
            "exam_type": row.get("exam_type"),
            "_date_obj": _parse_date(row.get("exam_date")),
        })
    parsed.sort(key=lambda r: (r["_date_obj"] is None, r["_date_obj"]))
    for r in parsed:
        r.pop("_date_obj", None)
    return parsed


def get_signals(student_id: str) -> list[dict]:
    """
    Signals from 'signal_sheet' tab.
    Columns: student_id | signal_type | severity | urgency | reason | timestamp | actioned
    Returns only signals that have NOT been actioned yet, newest first.
    """
    rows = _rows_by_student("signal_sheet", student_id)
    parsed = []
    for row in rows:
        actioned_raw = str(row.get("actioned", "")).strip().lower()
        actioned = actioned_raw in ("true", "yes", "1", "y")
        parsed.append({
            "signal_type": row.get("signal_type"),
            "severity": row.get("severity"),
            "urgency": row.get("urgency"),
            "reason": row.get("reason"),
            "timestamp": row.get("timestamp"),
            "actioned": actioned,
            "_date_obj": _parse_date(row.get("timestamp")),
        })
    # Newest first
    parsed.sort(key=lambda r: (r["_date_obj"] is None, r["_date_obj"]), reverse=True)
    active = [r for r in parsed if not r["actioned"]]
    for r in active:
        r.pop("_date_obj", None)
    return active


def invalidate_cache():
    """Call this if you need to force a fresh connection (e.g., after re-auth)."""
    global _client, _spreadsheet
    _client = None
    _spreadsheet = None