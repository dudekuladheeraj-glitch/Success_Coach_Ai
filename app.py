import os
import streamlit as st
from graph.graph_builder import graph
from services.sheets import get_roster
from services.knowledge_base import is_knowledge_base_ready
from services.memory import save_session_memory
from services.session_summary import extract_facts, generate_session_summary

st.set_page_config(page_title="Success Coach AI", page_icon="🎓", layout="wide")
st.title("🎓 Success Coach AI")


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "student_id" not in st.session_state:
    st.session_state.student_id = "STU001"


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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session Controls")
    options = load_student_options()
    labels = [label for label, _ in options]
    ids = [sid for _, sid in options]
    default_index = (
        ids.index(st.session_state.student_id)
        if st.session_state.student_id in ids
        else 0
    )
    selected_label = st.selectbox("Select Student", labels, index=default_index)
    selected_id = ids[labels.index(selected_label)]

    if selected_id != st.session_state.student_id:
        st.session_state.student_id = selected_id
        st.session_state.messages = []
        st.rerun()

    st.markdown(f"**Current student:** `{st.session_state.student_id}`")
    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔚 End Session"):
        try:
            # ── 1. Generate both memory types from this session's transcript ──
            with st.spinner("Generating session summary..."):
                summary = generate_session_summary(st.session_state.messages)

            with st.spinner("Extracting durable facts..."):
                facts = extract_facts(st.session_state.messages)

            st.subheader("📋 Session Summary")
            st.info(summary)

            if facts:
                st.subheader("🧠 Facts Learned About Student")
                st.info(facts)
            else:
                st.caption("No new durable facts identified in this session.")

            # ── 2. Persist both to mem0, tagged separately ────────────────────
            with st.spinner("Saving to Memory... (confirming indexing, may take a moment)"):
                result = save_session_memory(
                    student_id=st.session_state.student_id,
                    facts=facts,
                    summary=summary,
                )

            if result.get("summary_confirmed"):
                st.success("✅ Session summary saved and confirmed in memory")
            else:
                st.warning(
                    "⏳ Session summary was accepted by Mem0 but is still "
                    "processing in the background — it should appear shortly. "
                    "Wait a bit before asking a 'recap last session' question."
                )

            if facts:
                if result.get("fact_confirmed"):
                    st.success("✅ Facts saved and confirmed in memory")
                else:
                    st.warning(
                        "⏳ Facts were accepted by Mem0 but are still processing "
                        "in the background — check back in a moment."
                    )

            st.json(result)

            st.session_state.messages = []

        except Exception as e:
            st.error(f"Failed to save session memory: {e}")


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

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)