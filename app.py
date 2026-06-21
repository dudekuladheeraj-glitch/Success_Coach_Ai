import streamlit as st
from graph.graph_builder import graph
from services.sheets import get_roster
from services.memory import (
    save_session_memory,
    save_signal,
    get_all_active_signals_across_roster,
    mark_signal_actioned,
)
from services.session_summary import extract_facts, extract_signal, generate_session_summary
from services.daily_planner import generate_daily_plan
from services import calendar as cal

st.set_page_config(page_title="Success Coach AI", page_icon="🎓", layout="wide")
st.title("🎓 Success Coach AI")


# ── Google Calendar OAuth callback (M7) ───────────────────────────────────────
# Must run early, before any other widgets render, because Streamlit reruns
# the whole script top-to-bottom and the ?code=... param only appears once,
# right after Google redirects back here post-consent.

if "calendar_credentials" not in st.session_state:
    st.session_state.calendar_credentials = None

_query_params = st.query_params

if "code" in _query_params and not st.session_state.calendar_credentials:
    try:
        auth_code = _query_params["code"]

        # Exchange Google auth code for credentials using the PKCE verifier
        # stored earlier by cal.get_authorization_url()
        oauth_state = _query_params.get("state")
        _creds = cal.exchange_code_for_credentials(auth_code, oauth_state)
        st.session_state.calendar_credentials = cal.credentials_to_dict(_creds)

        # Clean temporary OAuth session values
        cal.clear_oauth_temp_state()

        # Drop ?code=... from the URL so refresh/rerun doesn't re-trigger token exchange
        st.query_params.clear()

        st.success("✅ Google Calendar connected successfully.")
        st.rerun()

    except Exception as exc:
        # Clear the URL params and temp state even on failure so the coach can retry cleanly
        cal.clear_oauth_temp_state()
        st.query_params.clear()
        st.error(f"Calendar connection failed: {exc}")

elif "error" in _query_params:
    # Coach declined consent on Google's screen, or some other OAuth error.
    _oauth_error = _query_params.get("error", "unknown_error")
    cal.clear_oauth_temp_state()
    st.query_params.clear()
    st.warning(f"Google Calendar connection was not completed ({_oauth_error}).")


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "student_id" not in st.session_state:
    st.session_state.student_id = "STU001"

# Guard against duplicate saves caused by Streamlit reruns
if "memory_saved_for_session" not in st.session_state:
    st.session_state.memory_saved_for_session = False

# Store end-session artifacts so they survive reruns
if "session_artifacts" not in st.session_state:
    st.session_state.session_artifacts = None

# M7: today's generated plan survives reruns until the coach regenerates it
if "daily_plan" not in st.session_state:
    st.session_state.daily_plan = None

if "calendar_results" not in st.session_state:
    st.session_state.calendar_results = None


# ── Roster ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_student_options() -> list[tuple[str, str]]:
    try:
        roster = get_roster()
        return [
            (f"{row['student_id']} — {row['name']}", row["student_id"])
            for row in roster
            if row.get("student_id")
        ]
    except Exception as exc:
        st.warning(f"Could not load roster from sheet: {exc}")
        return [
            ("STU001 — Arjun Kumar", "STU001"),
            ("STU002 — Priya Sharma", "STU002"),
            ("STU003 — Rahul Verma", "STU003"),
        ]


# ── Coach-facing alert panel (M6) ─────────────────────────────────────────────

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEVERITY_ICON  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


@st.cache_data(show_spinner=False, ttl=60)
def load_active_signals() -> list[dict]:
    try:
        student_ids = [sid for _, sid in load_student_options()]
        return get_all_active_signals_across_roster(student_ids)
    except Exception as exc:
        print(f"[alert_panel] Failed to load active signals: {exc}")
        return []


def render_alert_panel():
    signals = load_active_signals()
    if not signals:
        st.success("✅ No active alerts — nothing flagged across your roster.")
        return

    signals_sorted = sorted(
        signals,
        key=lambda s: (
            SEVERITY_ORDER.get(s.get("severity", "low"), 3),
            s.get("created_at") or "",
        ),
    )

    critical_or_high = [
        s for s in signals_sorted
        if s.get("severity") in ("critical", "high")
    ]
    if critical_or_high:
        st.warning(f"⚠️ {len(critical_or_high)} student(s) need attention today")

    for sig in signals_sorted:
        icon = SEVERITY_ICON.get(sig.get("severity", "low"), "⚪")
        with st.expander(
            f"{icon} {sig.get('student_id', '?')} — "
            f"{sig.get('signal_type', 'Signal')} "
            f"({sig.get('severity', 'unknown')})"
        ):
            st.markdown(f"**Urgency:** {sig.get('urgency', 'unknown')}")
            st.markdown(f"**Reason:** {sig.get('reason', '')}")
            st.caption(f"Flagged: {sig.get('created_at', '')}")

            if st.button("✅ Mark Actioned", key=f"action_{sig.get('id')}"):
                if mark_signal_actioned(sig.get("student_id"), sig.get("id")):
                    st.success("Marked actioned.")
                    load_active_signals.clear()
                    st.rerun()
                else:
                    st.error("Could not mark as actioned — see logs.")


# ── Coach-facing daily plan (M7) ──────────────────────────────────────────────


def render_calendar_connect():
    """Shows a connect link if the coach hasn't authorized Calendar access yet."""
    if cal.is_connected():
        st.success("✅ Google Calendar connected")
        if st.button("🔌 Disconnect Calendar"):
            st.session_state.calendar_credentials = None
            cal.clear_oauth_temp_state()
            st.rerun()
        return True

    st.info("Connect your Google Calendar to create events from today's plan.")
    try:
        auth_url = cal.get_authorization_url()
        st.link_button("🔗 Connect Google Calendar", auth_url)
    except ValueError as exc:
        st.error(f"Calendar isn't configured yet: {exc}")
    return False


def render_daily_plan():
    st.subheader("📅 Today's Plan")
    st.caption(
        "Pulls every un-actioned signal across the roster and builds a "
        "prioritized day — who needs attention, what kind of session, and why."
    )

    calendar_ready = render_calendar_connect()
    st.divider()

    if st.button("✨ Generate Today's Plan", type="primary"):
        with st.spinner("Reviewing flagged students and building today's schedule..."):
            try:
                st.session_state.daily_plan = generate_daily_plan()
                st.session_state.calendar_results = None  # clear stale results from a previous plan
            except Exception as exc:
                st.error(f"Failed to generate plan: {exc}")
                st.session_state.daily_plan = None

    plan = st.session_state.daily_plan
    if plan is None:
        return

    if plan.get("flagged_count", 0) == 0:
        st.success("✅ No students currently flagged — nothing to schedule today.")
        return

    if plan.get("parse_error"):
        st.warning(
            "⚠️ The plan came back in an unexpected format and couldn't be "
            "fully parsed. Showing whatever could be salvaged below — "
            "try regenerating if this looks incomplete."
        )

    assigned = plan.get("assigned_slots", [])
    deferred = plan.get("deferred", [])

    st.markdown(
        f"**{plan['flagged_count']} student(s) flagged** — "
        f"{len(assigned)} scheduled today, {len(deferred)} deferred."
    )

    if assigned:
        st.markdown("### Scheduled today")
        for i, slot in enumerate(assigned):
            icon = SEVERITY_ICON.get(slot.get("severity", "low"), "⚪")
            time_label = slot["time"].strftime("%I:%M %p")
            name = slot.get("student_name", slot.get("student_id", "Unknown"))
            session_type = slot.get("session_type", "Check-in")
            with st.expander(
                f"{time_label} — {icon} {name} — {session_type}",
                expanded=True,
            ):
                st.markdown(f"**Why:** {slot.get('plain_reason', '')}")
                st.caption(
                    f"Severity: {slot.get('severity', 'unknown')} · "
                    f"Urgency: {slot.get('urgency', 'unknown')} · "
                    f"Duration: {slot.get('duration_minutes', 30)} min"
                )

    if deferred:
        st.markdown("### Deferred to tomorrow")
        for item in deferred:
            icon = SEVERITY_ICON.get(item.get("severity", "low"), "⚪")
            name = item.get("student_name", item.get("student_id", "Unknown"))
            st.markdown(f"{icon} **{name}** — {item.get('defer_reason', '')}")

    # ── Calendar event creation ─────────────────────────────────────────────
    if assigned:
        st.divider()
        if not calendar_ready:
            st.caption("Connect Google Calendar above to create events for this plan.")
        else:
            if st.button("📆 Create Calendar Events for Today's Plan"):
                creds = cal.get_session_credentials()
                if not creds:
                    st.error("Calendar connection lost — please reconnect above.")
                else:
                    with st.spinner("Creating calendar events..."):
                        try:
                            results = cal.create_events_for_plan(creds, assigned)
                            st.session_state.calendar_results = results
                        except Exception as exc:
                            st.error(f"Failed to create calendar events: {exc}")

    results = st.session_state.get("calendar_results")
    if results:
        succeeded = [r for r in results if r["success"]]
        failed    = [r for r in results if not r["success"]]
        if succeeded:
            st.success(f"✅ {len(succeeded)} event(s) created on your calendar.")
            for r in succeeded:
                if r.get("event_link"):
                    st.caption(f"• [{r['student_id']}]({r['event_link']})")
        if failed:
            st.error(f"⚠️ {len(failed)} event(s) failed to create — see details:")
            for r in failed:
                st.caption(f"• {r['student_id']}: {r.get('error', 'unknown error')}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🚨 Alerts")
    render_alert_panel()
    if st.button("🔄 Refresh Alerts"):
        load_active_signals.clear()
        st.rerun()
    st.divider()

    st.header("Session Controls")
    options = load_student_options()
    labels  = [label for label, _ in options]
    ids     = [sid   for _, sid   in options]
    default_index = (
        ids.index(st.session_state.student_id)
        if st.session_state.student_id in ids
        else 0
    )
    selected_label = st.selectbox("Select Student", labels, index=default_index)
    selected_id    = ids[labels.index(selected_label)]

    if selected_id != st.session_state.student_id:
        st.session_state.student_id            = selected_id
        st.session_state.messages              = []
        st.session_state.memory_saved_for_session = False
        st.session_state.session_artifacts     = None
        st.rerun()

    st.markdown(f"**Current student:** `{st.session_state.student_id}`")
    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages              = []
        st.session_state.memory_saved_for_session = False
        st.session_state.session_artifacts     = None
        st.rerun()

    # ── End Session ───────────────────────────────────────────────────────────
    if st.button("🔚 End Session"):

        # ── Guard: only run once per session ──────────────────────────────────
        if not st.session_state.memory_saved_for_session:
            st.session_state.memory_saved_for_session = True

            try:
                messages_snapshot = list(st.session_state.messages)

                # ── 1. Generate artifacts ──────────────────────────────────────
                with st.spinner("Generating session summary..."):
                    summary = generate_session_summary(messages_snapshot)

                with st.spinner("Extracting durable facts..."):
                    facts = extract_facts(messages_snapshot)

                with st.spinner("Checking for any concerning signals..."):
                    signal = extract_signal(messages_snapshot)

                # ── 2. Save to Mem0 (once) ─────────────────────────────────────
                with st.spinner("Saving to memory..."):
                    mem_result = save_session_memory(
                        student_id=st.session_state.student_id,
                        facts=facts,
                        summary=summary,
                    )

                signal_saved = False
                if signal:
                    try:
                        save_signal(st.session_state.student_id, signal)
                        signal_saved = True
                        load_active_signals.clear()
                    except Exception as exc:
                        print(f"[end_session] save_signal failed: {exc}")

                # ── 3. Store artifacts in session state for display ────────────
                st.session_state.session_artifacts = {
                    "summary":      summary,
                    "facts":        facts,
                    "signal":       signal,
                    "signal_saved": signal_saved,
                    "mem_result":   mem_result,
                }

                # ── 4. Clear messages ──────────────────────────────────────────
                st.session_state.messages = []

            except Exception as e:
                st.error(f"Failed to end session: {e}")
                st.session_state.memory_saved_for_session = False

    # ── Display artifacts (survives reruns) ───────────────────────────────────
    artifacts = st.session_state.get("session_artifacts")
    if artifacts:
        st.subheader("📋 Session Summary")
        st.info(artifacts["summary"])

        if artifacts["facts"]:
            st.subheader("🧠 Facts Learned About Student")
            st.info(artifacts["facts"])
        else:
            st.caption("No new durable facts identified in this session.")

        signal = artifacts["signal"]
        if signal:
            icon = SEVERITY_ICON.get(signal["severity"], "⚪")
            st.subheader(f"{icon} Signal Flagged")
            st.warning(
                f"**{signal['signal_type']}** — severity: `{signal['severity']}`, "
                f"urgency: `{signal['urgency']}`\n\n{signal['reason']}"
            )
            if artifacts["signal_saved"]:
                st.success("✅ Signal saved to memory")
            else:
                st.error("⚠️ Signal detected but failed to save — check logs.")
        else:
            st.caption("No concerning signal identified in this session.")

        mem_result = artifacts["mem_result"]
        if mem_result.get("summary_confirmed"):
            st.success("✅ Session summary saved and confirmed in memory")
        else:
            st.warning(
                "⏳ Session summary accepted by Mem0 — still processing in "
                "the background. Wait a moment before asking 'recap last session'."
            )

        if artifacts["facts"]:
            if mem_result.get("fact_confirmed"):
                st.success("✅ Facts saved and confirmed in memory")
            else:
                st.warning(
                    "⏳ Facts accepted by Mem0 — still processing in the background."
                )

        # Reset button so coach can start a new session
        if st.button("🆕 Start New Session"):
            st.session_state.memory_saved_for_session = False
            st.session_state.session_artifacts        = None
            st.rerun()


# ── Main content: Chat (student-facing) vs Today's Plan (coach-facing) ────────

tab_chat, tab_plan = st.tabs(["💬 Chat", "📅 Today's Plan"])
with tab_chat:
    # Chat input first so the newest message gets added to session state
    message = st.chat_input("Ask your coach anything…")

    if message:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": message})

        # Generate assistant reply
        with st.spinner("Thinking…"):
            try:
                history = st.session_state.messages[:-1][-20:]
                result = graph.invoke(
                    {
                        "message": message,
                        "student_id": st.session_state.student_id,
                        "intent": "",
                        "student_context": {},
                        "alerts": [],
                        "kb_context": "",
                        "response": "",
                        "chat_history": history,
                        "known_facts": [],
                        "session_summaries": [],
                    }
                )
                response = result["response"]
            except Exception as error:
                response = f"⚠️ Something went wrong.\n\n**Error:** `{error}`"

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Rerun so the full chat history redraws cleanly above the input
        st.rerun()

    # Render full chat history AFTER processing any new message
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


with tab_plan:
    render_daily_plan()