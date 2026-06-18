from llm.openai_client import get_llm

llm = get_llm()

def chat_node(state):

    response = llm.invoke(
        state["message"]
    )

    return {
        "response": response.content
    }

def load_student