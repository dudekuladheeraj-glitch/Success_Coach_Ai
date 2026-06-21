"""
services/calendar.py
─────────────────────
Google Calendar integration for the coach's OWN calendar (M7).

This module uses OAuth Authorization Code flow with PKCE.
IMPORTANT: We persist the PKCE code_verifier inside the OAuth `state`
parameter so the flow still works even if Streamlit creates a fresh
session after redirecting back from Google.
"""

import os
import json
import base64
import secrets
from datetime import datetime, timedelta

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _get_env(key: str) -> str:
    """Read from st.secrets first (Streamlit Cloud), then environment (.env/local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, "")


def _client_config() -> tuple[dict, str]:
    client_id = _get_env("GOOGLE_CALENDAR_CLIENT_ID")
    client_secret = _get_env("GOOGLE_CALENDAR_CLIENT_SECRET")
    redirect_uri = _get_env("GOOGLE_CALENDAR_REDIRECT_URI")

    missing = [
        name for name, val in [
            ("GOOGLE_CALENDAR_CLIENT_ID", client_id),
            ("GOOGLE_CALENDAR_CLIENT_SECRET", client_secret),
            ("GOOGLE_CALENDAR_REDIRECT_URI", redirect_uri),
        ] if not val
    ]

    if missing:
        raise ValueError(
            f"Missing calendar config: {', '.join(missing)}. "
            "Set these in .env (local) or Streamlit secrets (cloud)."
        )

    config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    return config, redirect_uri


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for encoding / decoding OAuth state
# ──────────────────────────────────────────────────────────────────────────────

def _encode_state(payload: dict) -> str:
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _decode_state(state: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(state.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid OAuth state payload: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# OAuth flow
# ──────────────────────────────────────────────────────────────────────────────

def get_authorization_url() -> str:
    """
    Build the Google consent URL.

    We store the PKCE code_verifier INSIDE the OAuth state parameter so the
    callback can recover it even if Streamlit starts a fresh session.
    """
    config, redirect_uri = _client_config()

    code_verifier = secrets.token_urlsafe(64)

    # Put the verifier inside OAuth state so we can recover it after redirect
    state_payload = {
        "calendar_pkce_verifier": code_verifier,
        "nonce": secrets.token_urlsafe(16),
    }
    encoded_state = _encode_state(state_payload)

    flow = Flow.from_client_config(
        config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        autogenerate_code_verifier=False,
    )

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
        state=encoded_state,
    )
    return auth_url


def exchange_code_for_credentials(auth_code: str, state: str | None) -> Credentials:
    """
    Exchange the returned ?code=... for access/refresh tokens using the SAME
    PKCE verifier that was embedded in the OAuth state parameter.
    """
    config, redirect_uri = _client_config()

    if not state:
        raise ValueError(
            "Missing OAuth state in callback. Please click "
            "'Connect Google Calendar' and try again."
        )

    state_payload = _decode_state(state)
    code_verifier = state_payload.get("calendar_pkce_verifier")
    if not code_verifier:
        raise ValueError(
            "Missing PKCE verifier in OAuth state. Please click "
            "'Connect Google Calendar' and try again."
        )

    flow = Flow.from_client_config(
        config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        autogenerate_code_verifier=False,
    )

    flow.fetch_token(code=auth_code)
    return flow.credentials


def credentials_to_dict(creds: Credentials) -> dict:
    """Convert Credentials object to a serializable dict for session_state."""
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }


def credentials_from_dict(data: dict) -> Credentials:
    """Rebuild Credentials object from dict stored in session_state."""
    return Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Calendar API helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_service(creds: Credentials):
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def is_connected() -> bool:
    """True if session_state has stored calendar credentials."""
    try:
        import streamlit as st
        return bool(st.session_state.get("calendar_credentials"))
    except Exception:
        return False


def get_session_credentials() -> Credentials | None:
    """Load credentials from session_state if present."""
    try:
        import streamlit as st
        data = st.session_state.get("calendar_credentials")
    except Exception:
        data = None

    if not data:
        return None
    return credentials_from_dict(data)


def clear_oauth_temp_state():
    """
    Kept for compatibility with app.py. We no longer need temp PKCE state in
    session because verifier is stored inside OAuth state, but this can remain
    as a harmless no-op.
    """
    try:
        import streamlit as st
        st.session_state.pop("calendar_code_verifier", None)
        st.session_state.pop("calendar_oauth_state", None)
    except Exception:
        pass


def create_event(
    creds: Credentials,
    summary: str,
    description: str,
    start_time: datetime,
    duration_minutes: int = 30,
    calendar_id: str = "primary",
) -> dict:
    """
    Create a single event on the coach's calendar.
    """
    service = _get_service(creds)
    end_time = start_time + timedelta(minutes=duration_minutes)

    event_body = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
    }

    return service.events().insert(calendarId=calendar_id, body=event_body).execute()


def create_events_for_plan(creds: Credentials, plan_items: list[dict]) -> list[dict]:
    """
    Batch-create calendar events for each assigned item in the daily plan.
    """
    results = []

    for item in plan_items:
        student_id = item.get("student_id", "unknown")
        try:
            student_name = item.get("student_name", student_id)
            session_type = item.get("session_type", "Check-in")

            event = create_event(
                creds=creds,
                summary=f"Coaching: {student_name} — {session_type}",
                description=item.get("reason") or item.get("plain_reason", ""),
                start_time=item["time"],
                duration_minutes=item.get("duration_minutes", 30),
            )

            results.append({
                "student_id": student_id,
                "success": True,
                "event_link": event.get("htmlLink"),
                "error": None,
            })

        except HttpError as exc:
            results.append({
                "student_id": student_id,
                "success": False,
                "event_link": None,
                "error": str(exc),
            })

        except Exception as exc:
            results.append({
                "student_id": student_id,
                "success": False,
                "event_link": None,
                "error": str(exc),
            })

    return results