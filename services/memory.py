import os

from mem0 import MemoryClient


_memory_client = None


def get_memory_client():
    global _memory_client

    if _memory_client is not None:
        return _memory_client

    api_key = os.getenv("MEM0_API_KEY")

    if not api_key:
        raise ValueError(
            "MEM0_API_KEY is not set."
        )

    _memory_client = MemoryClient(
        api_key=api_key
    )

    return _memory_client


def save_session_summary(
    student_id: str,
    summary: str,
):
    client = get_memory_client()

    messages = [
        {
            "role": "user",
            "content": summary,
        }
    ]

    return client.add(
        messages,
        user_id=student_id,
    )