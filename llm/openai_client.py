from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
         model="gpt-5.4-mini-2026-03-17",
        temperature=0.3,
    )
