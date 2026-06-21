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

st.set_page_config(page_title="Success Coach AI", page_icon="🎓", layout="wide")
st.title("🎓 Success Coach AI")


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


# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────

message = st.chat_input("Ask your coach anything…")

if message:
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    with st.spinner("Thinking…"):
        try:
            history = st.session_state.messages[:-1][-20:]
            result  = graph.invoke(
                {
                    "message":         message,
                    "student_id":      st.session_state.student_id,
                    "intent":          "",
                    "student_context": {},
                    "alerts":          [],
                    "kb_context":      "",
                    "response":        "",
                    "chat_history":    history,
                    "known_facts":     [],
                    "session_summaries": [],
                }
            )
            response = result["response"]
        except Exception as error:
            response = f"⚠️ Something went wrong.\n\n**Error:** `{error}`"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)