import streamlit as st

from graph.graph_builder import graph
from services.sheets import get_roster

st.set_page_config(
    page_title="Success Coach AI",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Success Coach AI")

# ── Session state initialisation ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "student_id" not in st.session_state:
    st.session_state.student_id = "STU001"


# ── Load student roster for the sidebar dropdown ──────────────────────────────

@st.cache_data(show_spinner=False)
def load_student_options() -> list[tuple[str, str]]:
    """Returns list of (display_label, student_id) tuples."""
    try:
        roster = get_roster()
        return [
            (f"{row['student_id']} — {row['name']}", row["student_id"])
            for row in roster
            if row.get("student_id")
        ]
    except Exception as exc:
        st.warning(f"Could not load roster from sheet: {exc}")
        # Fallback so the app still runs during local dev / auth setup
        # return [
        #     ("STU001 — Arjun Kumar", "STU001"),
        #     ("STU002 — Priya Sharma", "STU002"),
        #     ("STU003 — Rahul Verma", "STU003"),
        # ]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session Controls")

    options = load_student_options()
    labels = [label for label, _ in options]
    ids = [sid for _, sid in options]

    default_index = ids.index(st.session_state.student_id) if st.session_state.student_id in ids else 0

    selected_label = st.selectbox("Select Student", labels, index=default_index)
    selected_id = ids[labels.index(selected_label)]

    # Reset chat when a different student is selected
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
        st.session_state.clear()
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

message = st.chat_input("Ask your coach anything…")

if message:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    # Run the LangGraph pipeline
    with st.spinner("Thinking…"):
        try:
            # Pass everything said BEFORE this turn (excludes the message
            # just appended above). Capped to the last 20 turns to keep the
            # prompt size reasonable on long sessions.
            history = st.session_state.messages[:-1][-20:]

            result = graph.invoke(
                {
                    "message": message,
                    "student_id": st.session_state.student_id,
                    "intent": "",
                    "student_context": {},
                    "alerts": [],
                    "response": "",
                    "chat_history": history,
                }
            )
            response = result["response"]
        except Exception as error:
            response = (
                f"⚠️ Sorry, something went wrong while fetching your data.\n\n"
                f"**Error:** `{error}`\n\n"
                "Please check that:\n"
                "- `credentials.json` is in the project root\n"
                "- `GOOGLE_SPREADSHEET_ID` is set in `.env`\n"
                "- The spreadsheet is shared with your Google account"
            )

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)