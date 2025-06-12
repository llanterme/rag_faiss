"""RetrievalQA chain configuration module."""

from typing import Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferMemory  # Will show deprecation warning but still works
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from src.config import settings


def build_retrieval_chain(
    index: FAISS, model_name: str = "gpt-4o"
) -> BaseConversationalRetrievalChain:
    """Build a retrieval chain for question answering.

    Args:
        index: FAISS vector store to use for retrieval.
        model_name: Name of the OpenAI model to use.

    Returns:
        Configured ConversationalRetrievalChain.
    """
    # Create memory for conversation context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=settings.temperature,
        openai_api_key=settings.openai_api_key,
    )

    # Create the retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.as_retriever(),
        memory=memory,
        verbose=False,
    )

    return chain
