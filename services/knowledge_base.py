# """
# services/knowledge_base.py
# ──────────────────────────
# Chroma Cloud-backed knowledge base.
# """

# import os

# import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# from dotenv import load_dotenv

# load_dotenv()

# COLLECTION_NAME = "knowledge_base"
# TOP_K = 4

# _client = None
# _collection = None


# def _get_collection():
#     global _client, _collection

#     if _collection is not None:
#         return _collection

#     _client = chromadb.CloudClient(
#         api_key=os.getenv("CHROMA_API_KEY"),
#         tenant=os.getenv("CHROMA_TENANT"),
#         database=os.getenv("CHROMA_DATABASE"),
#     )

#     embed_fn = OpenAIEmbeddingFunction(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model_name="text-embedding-3-small",
#     )

#     _collection = _client.get_collection(
#         name=COLLECTION_NAME,
#         embedding_function=embed_fn,
#     )

#     return _collection


# def query_knowledge_base(
#     question: str,
#     top_k: int = TOP_K,
# ) -> str:

#     try:
#         collection = _get_collection()

#         results = collection.query(
#             query_texts=[question],
#             n_results=top_k,
#         )

#         docs = results.get("documents", [[]])[0]

#         if not docs:
#             return ""

#         return "\n\n---\n\n".join(docs)

#     except Exception as exc:
#         print(f"[knowledge_base] query failed: {exc}")
#         return ""


# def is_knowledge_base_ready() -> bool:

#     try:
#         collection = _get_collection()
#         return collection.count() > 0

#     except Exception as exc:
#         print(f"[knowledge_base] readiness check failed: {exc}")
#         return False


"""
services/knowledge_base.py
──────────────────────────
Chroma Cloud-backed knowledge base.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "knowledge_base"
TOP_K = 4

_client     = None
_collection = None


def _get_env(key: str) -> str:
    """Read from st.secrets first, then .env"""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")



def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.CloudClient(
        api_key=_get_env("CHROMA_API_KEY"),
        tenant=_get_env("CHROMA_TENANT"),
        database=_get_env("CHROMA_DATABASE"),
    )

    embed_fn = OpenAIEmbeddingFunction(
        api_key=_get_env("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return _collection


def query_knowledge_base(question: str, top_k: int = TOP_K) -> str:
    try:
        collection = _get_collection()
        results    = collection.query(query_texts=[question], n_results=top_k)
        docs       = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n---\n\n".join(docs)
    except Exception as exc:
        print(f"[knowledge_base] query failed: {exc}")
        return ""


def is_knowledge_base_ready() -> bool:
    try:
        collection = _get_collection()
        return collection.count() > 0
    except Exception as exc:
        print(f"[knowledge_base] readiness check failed: {exc}")
        return False






