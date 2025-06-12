"""LangGraph-based retrieval chain module for document chat.

This module implements retrieval functionality using LangGraph for improved
conversation memory management and state persistence.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.config import settings


class ChatState(TypedDict):
    """State representation for the RAG chat graph.

    Attributes:
        messages: List of conversation messages (HumanMessage or AIMessage)
        question: Current user question
        context: Retrieved context from the vector store as a combined string
        answer: Generated AI response
    """

    messages: List[Any]  # HumanMessage or AIMessage
    question: Optional[str]
    context: Optional[str]  # Using string for context, not List[str]
    answer: Optional[str]


def create_graph_chain(
    index: FAISS, model_name: str = "gpt-4o"
) -> StateGraph:
    """Create a LangGraph-based retrieval QA chain.

    This replaces the deprecated ConversationalRetrievalChain with a modern
    LangGraph implementation that provides better conversation state management.

    Args:
        index: FAISS vector store for retrieval
        model_name: Name of the OpenAI model to use

    Returns:
        A configured StateGraph for chat interaction
    """
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=settings.temperature,
        openai_api_key=settings.openai_api_key,
    )

    # Define retrieval function
    def retrieve(state: ChatState) -> ChatState:
        """Retrieve relevant documents from the vector store.

        Args:
            state: Current conversation state

        Returns:
            Updated state with retrieved context
        """
        retriever = index.as_retriever()
        if not state.get("question"):
            return state
        
        try:
            # Use invoke method instead of deprecated get_relevant_documents
            context = retriever.invoke(state["question"])
            context_str = "\n\n".join([doc.page_content for doc in context])
            return {**state, "context": context_str}
        except Exception as e:
            print(f"Retrieval error: {e}")
            # Return empty context if retrieval fails
            return {**state, "context": ""}

    # Define answer generation function
    def generate_answer(state: ChatState) -> ChatState:
        """Generate an answer using the LLM."""
        if not state.get("question") or state.get("context") is None:
            return state

        # Create the prompt with conversation history and retrieved context
        messages = [
            SystemMessage(
                content=f"You are a helpful AI assistant answering questions based on the provided context.\n\nContext: {state['context']}"
            ),
        ]

        # Add conversation history
        if state.get("messages"):
            messages.extend(state["messages"])

        # Add the current question
        messages.append(HumanMessage(content=state["question"]))

        # Generate response from LLM
        try:
            response = llm.invoke(messages)
            return {**state, "answer": response.content}
        except Exception as e:
            print(f"Answer generation error: {e}")
            return {**state, "answer": "I encountered an error while generating an answer. Please try again."}

    # Define function to update conversation history
    def update_conversation(state: ChatState) -> ChatState:
        """Update conversation state with the new QA pair."""
        if not state.get("question") or not state.get("answer"):
            return state

        # Initialize messages list if it doesn't exist
        messages = state.get("messages", [])
        
        # Add the user question and generated answer to conversation history
        new_messages = messages.copy()
        new_messages.append(HumanMessage(content=state["question"]))
        new_messages.append(AIMessage(content=state["answer"]))
        
        # Create new state with updated messages and cleared question/answer/context
        return {
            **state,
            "messages": new_messages,
            "question": None,
            "answer": None,
            "context": None
        }

    # Create the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("update", update_conversation)
    
    # Set the entry point
    workflow.set_entry_point("retrieve")
    
    # Define the edges (execution flow)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "update")
    workflow.add_edge("update", END)
    
    # Compile the graph
    return workflow.compile()


def build_retrieval_chain(
    index: FAISS, model_name: str = "gpt-4o"
) -> Any:
    """Build a retrieval chain for question answering.

    This is a factory function that creates a LangGraph-based retrieval chain
    with integrated conversation memory.

    Args:
        index: FAISS vector store to use for retrieval.
        model_name: Name of the OpenAI model to use.

    Returns:
        Configured LangGraph chain with conversation memory.
    """
    # Create graph-based chain
    chain = create_graph_chain(index, model_name)

    # Initialize with empty conversation
    initial_state = {
        "messages": [],
        "question": None,
        "context": None,
        "answer": None,
    }
    
    # Ensure the graph state directory exists
    settings.graph_state_path.mkdir(parents=True, exist_ok=True)
    
    # In LangGraph 0.4.x, we need to first compile the graph then attach persistence
    compiled_graph = chain
    
    # Set up persistence using the new API
    if settings.enable_persistence:
        try:
            from langgraph.checkpoint import SqliteCheckpoint
            # Use SQLite for persistent storage
            persistence_path = str(settings.graph_state_path / "graph_state.sqlite")
            
            # Create a checkpoint instance
            checkpoint = SqliteCheckpoint(persistence_path)
            
            # Add persistence to the graph
            compiled_graph = compiled_graph.with_state_checkpoint(checkpoint)
            
        except (ImportError, AttributeError):
            # Fall back to memory-only if SQLite support not available or API mismatch
            pass
    
    # Return the chain (with persistence if enabled)
    return compiled_graph
