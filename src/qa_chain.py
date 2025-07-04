"""Legacy module that redirects to langgraph_chain implementation.

This module is maintained for backwards compatibility but delegates to the new
LangGraph-based implementation.
"""

from typing import Any

from langchain_community.vectorstores import FAISS

from src.langgraph_chain import build_retrieval_chain as _build_retrieval_chain


def build_retrieval_chain(index: FAISS, model_name: str = "gpt-4o") -> Any:
    """Build a retrieval chain for question answering.

    This is a wrapper around the LangGraph implementation that maintains
    backwards compatibility with the original API.

    Args:
        index: FAISS vector store to use for retrieval.
        model_name: Name of the OpenAI model to use.

    Returns:
        Configured LangGraph chain with conversation memory.
    """
    # Delegate to the new LangGraph implementation
    return _build_retrieval_chain(index, model_name)
