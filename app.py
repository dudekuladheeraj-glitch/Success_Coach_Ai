import streamlit as st
from graph.graph_builder import graph

st.set_page_config(
    page_title="Success Coach AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Success Coach AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Session Controls")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔚 End Session"):
        st.session_state.clear()
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
message = st.chat_input("Type your message...")

if message:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    with st.chat_message("user"):
        st.markdown(message)

    # Call LangGraph
    with st.spinner("Thinking..."):

        result = graph.invoke(
            {
                "message": message
            }
        )

        response = result["response"]

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)