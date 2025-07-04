"""LangGraph-based retrieval chain module for document chat.

This module implements retrieval functionality using LangGraph for improved
conversation memory management and state persistence.
"""

import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.config import LLMProvider, settings
from src.observability import logfire_manager, log_operation
from src.prompts import create_rag_prompt, PromptStyle

try:
    import logfire
except ImportError:
    logfire = None


class ChatState(TypedDict):
    """State representation for the RAG chat graph.

    Attributes:
        messages: List of conversation messages (HumanMessage or AIMessage)
        question: Current user question
        context: Retrieved context from the vector store as a combined string
        sources: List of document sources for the retrieved context
        answer: Generated AI response
        student_id: Optional student identifier for context
    """

    messages: List[Any]  # HumanMessage or AIMessage
    question: Optional[str]
    context: Optional[str]  # Using string for context, not List[str]
    sources: Optional[List[Dict[str, str]]]  # Document sources with metadata
    answer: Optional[str]
    student_id: Optional[str]  # Track student context


def get_llm(model_name: Optional[str] = None) -> BaseChatModel:
    """Get the appropriate LLM based on settings.

    Args:
        model_name: Optional model name to override the one in settings

    Returns:
        Chat model instance based on the configured provider
    """
    # Use the model name from settings if not provided
    model = model_name or settings.llm_model

    if settings.llm_provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model_name=model,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == LLMProvider.OLLAMA:
        # Default to llama3.2 if using Ollama without a specific model
        if not model or model == "gpt-4o":
            model = "llama3.2:latest"

        return ChatOllama(
            model=model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_graph_chain(index: FAISS, model_name: Optional[str] = None, student_id: Optional[str] = None) -> StateGraph:
    """Create a LangGraph-based retrieval QA chain.

    This replaces the deprecated ConversationalRetrievalChain with a modern
    LangGraph implementation that provides better conversation state management.

    Args:
        index: FAISS vector store for retrieval
        model_name: Optional name of the model to use (overrides settings)
        student_id: Optional student identifier for context tracking

    Returns:
        A configured StateGraph for chat interaction
    """
    # Initialize the language model based on the configured provider
    llm = get_llm(model_name)

    # Define retrieval function
    @log_operation("retrieve_documents")
    def retrieve(state: ChatState) -> ChatState:
        """Retrieve relevant documents from the vector store.

        Args:
            state: Current conversation state

        Returns:
            Updated state with retrieved context and sources
        """
        if not state.get("question"):
            print("No question in state, skipping retrieval")
            return state

        # Validate question
        question = state["question"]
        if not question or not isinstance(question, str) or len(question.strip()) < 3:
            print(f"Invalid question format: {question}")
            return {**state, "context": "", "sources": []}

        start_time = time.time()

        try:
            with logfire_manager.span("document_retrieval") as span:
                span.set_attribute(
                    "query", question[:200] + "..." if len(question) > 200 else question
                )
                span.set_attribute("query_length", len(question))
                if state.get("student_id"):
                    span.set_attribute("student_id", state["student_id"])

                # Configure retriever with proper settings
                retriever = index.as_retriever(
                    search_kwargs={"k": 4}  # Retrieve top 4 documents
                )

                # Use invoke method instead of deprecated get_relevant_documents
                docs = retriever.invoke(question)

                span.set_attribute("documents_retrieved", len(docs) if docs else 0)

                if not docs:
                    print("No documents retrieved for the question")
                    return {**state, "context": "", "sources": []}

            # Extract document content and source information
            contents = []
            for doc in docs:
                if hasattr(doc, "page_content") and doc.page_content:
                    contents.append(doc.page_content)

            # If no valid content was found
            if not contents:
                print("Retrieved documents had no valid content")
                return {**state, "context": "", "sources": []}

            context_str = "\n\n".join(contents)

            # Extract source information from metadata
            sources = []
            for doc in docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    source_info = {
                        "source": doc.metadata.get("source", "Unknown"),
                        "source_path": doc.metadata.get("source_path", ""),
                    }
                    if source_info not in sources:
                        sources.append(source_info)

            print(f"Retrieved {len(docs)} documents with {len(sources)} unique sources")
            return {**state, "context": context_str, "sources": sources}
        except Exception as e:
            import traceback

            print(f"Retrieval error: {e}")
            print(traceback.format_exc())
            # Return empty context if retrieval fails
            return {**state, "context": "", "sources": []}

    # Define answer generation function
    @log_operation("generate_answer")
    def generate_answer(state: ChatState) -> ChatState:
        """Generate an answer using the LLM."""
        if not state.get("question"):
            print("No question found in state")
            return {**state, "answer": "No question was provided."}

        if state.get("context") is None:
            print("No context found in state")
            return {**state, "answer": "No relevant documents found for this question."}

        # Check if context is empty
        if state.get("context") == "":
            print("Empty context string")
            return {
                **state,
                "answer": "I couldn't find any relevant information to answer your question.",
            }

        start_time = time.time()

        with logfire_manager.span("llm_generation") as span:
            span.set_attribute(
                "question",
                (
                    state["question"][:200] + "..."
                    if len(state["question"]) > 200
                    else state["question"]
                ),
            )
            span.set_attribute("context_length", len(state.get("context", "")))
            span.set_attribute("llm_provider", str(settings.llm_provider))
            span.set_attribute("llm_model", settings.llm_model)

            # Create enhanced prompt with better structure and instructions
            enhanced_prompt = create_rag_prompt(
                context=state["context"],
                question=state["question"],
                sources=state.get("sources", []),
                style=getattr(settings, "prompt_style", "default"),
            )

            messages = [
                SystemMessage(content=enhanced_prompt),
            ]

            # Add conversation history (validate before adding)
            if state.get("messages") and isinstance(state["messages"], list):
                # Only add valid messages
                valid_messages = []
                for msg in state["messages"]:
                    if hasattr(msg, "content") and msg.content is not None:
                        valid_messages.append(msg)
                messages.extend(valid_messages)

            # Add the current question
            messages.append(HumanMessage(content=state["question"]))

            span.set_attribute("message_count", len(messages))
            span.set_attribute(
                "total_input_length",
                sum(len(getattr(msg, "content", "")) for msg in messages),
            )

            # Log the full prompt being sent to the LLM (if enabled)
            if settings.logfire_log_prompts:
                prompt_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        prompt_messages.append(f"[SYSTEM]: {msg.content}")
                    elif isinstance(msg, HumanMessage):
                        prompt_messages.append(f"[HUMAN]: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        prompt_messages.append(f"[AI]: {msg.content}")

                full_prompt = "\n\n".join(prompt_messages)
                span.set_attribute(
                    "full_prompt",
                    (
                        full_prompt[:1000] + "..."
                        if len(full_prompt) > 1000
                        else full_prompt
                    ),
                )

                # Also log to console for immediate visibility
                if logfire:
                    logfire.info(
                        "LLM Prompt",
                        prompt_preview=(
                            full_prompt[:500] + "..."
                            if len(full_prompt) > 500
                            else full_prompt
                        ),
                        prompt_length=len(full_prompt),
                        message_breakdown={
                            "system_messages": sum(
                                1 for m in messages if isinstance(m, SystemMessage)
                            ),
                            "human_messages": sum(
                                1 for m in messages if isinstance(m, HumanMessage)
                            ),
                            "ai_messages": sum(
                                1 for m in messages if isinstance(m, AIMessage)
                            ),
                        },
                    )
                else:
                    # Fallback to print if logfire not available
                    print(
                        f"\n=== LLM PROMPT ===\n{full_prompt[:500]}{'...' if len(full_prompt) > 500 else ''}\n=================\n"
                    )

            # Generate response from LLM
            try:
                # print(f"Calling OpenAI API with {len(messages)} messages")
                response = llm.invoke(messages)
                response_time = time.time() - start_time

                if response and hasattr(response, "content") and response.content:
                    span.set_attribute("response_length", len(response.content))
                    span.set_attribute("response_time_seconds", response_time)

                    # Log the complete interaction
                    logfire_manager.log_query(
                        query=state["question"],
                        response=response.content,
                        sources=state.get("sources", []),
                        response_time=response_time,
                    )

                    return {**state, "answer": response.content}
                else:
                    print("Empty or invalid response from LLM")
                    span.set_attribute("error", "empty_response")
                    return {
                        **state,
                        "answer": "I received an empty response. Please try rephrasing your question.",
                    }
            except Exception as e:
                import traceback

                print(f"Answer generation error: {str(e)}")
                print(traceback.format_exc())

                span.set_attribute("error", str(e))
                span.set_attribute("error_type", type(e).__name__)

                logfire_manager.log_error(e, "llm_generation")

                error_msg = f"Error: {str(e)}"
                # For token limit errors, provide a more helpful message
                if (
                    "maximum context length" in str(e).lower()
                    or "token" in str(e).lower()
                ):
                    error_msg = "The context from your documents is too large for me to process. Try uploading shorter documents or ask a more specific question."
                return {**state, "answer": error_msg}

    # Define function to update conversation history
    def update_conversation(state: ChatState) -> ChatState:
        """Update conversation state with the new QA pair."""
        if not state.get("question") or not state.get("answer"):
            return state

        # Initialize messages list if it doesn't exist
        messages = state.get("messages", [])

        # Get the sources from the state
        sources = state.get("sources", [])

        # Add the user question as a HumanMessage
        new_messages = messages.copy()
        new_messages.append(HumanMessage(content=state["question"]))

        # Create an AIMessage with the answer and attach sources as an attribute
        ai_message = AIMessage(content=state["answer"])
        if hasattr(ai_message, "__setattr__"):
            ai_message.sources = sources
        new_messages.append(ai_message)

        # Create new state with updated messages and preserve answer for final result
        # Keep the answer in the final state so it can be returned to the user
        return {
            **state,
            "messages": new_messages,
            "question": None,
            "answer": state["answer"],  # Preserve the answer for the final result
            "context": None,
            "sources": sources,  # Keep sources for reference
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


def get_student_graph_state_path(student_id: Optional[str] = None) -> Path:
    """Get graph state path for student or default."""
    if student_id:
        return settings.graph_state_path.parent / "students" / student_id / "graph_state"
    return settings.graph_state_path


def build_retrieval_chain(index: FAISS, model_name: Optional[str] = None, student_id: Optional[str] = None) -> Any:
    """Build a retrieval chain for question answering.

    This is a factory function that creates a LangGraph-based retrieval chain
    with integrated conversation memory.

    Args:
        index: FAISS vector store to use for retrieval.
        model_name: Name of the OpenAI model to use.
        student_id: Optional student identifier for isolated conversation state.

    Returns:
        Configured LangGraph chain with conversation memory.
    """
    # Create graph-based chain
    chain = create_graph_chain(index, model_name, student_id)

    # Initialize with empty conversation
    initial_state = {
        "messages": [],
        "question": None,
        "context": None,
        "sources": [],
        "answer": None,
        "student_id": student_id,
    }

    # Ensure the graph state directory exists
    graph_path = get_student_graph_state_path(student_id)
    graph_path.mkdir(parents=True, exist_ok=True)

    # In LangGraph 0.4.x, we need to first compile the graph then attach persistence
    compiled_graph = chain

    # Set up persistence using the new API
    if settings.enable_persistence:
        try:
            from langgraph.checkpoint import SqliteCheckpoint

            # Use SQLite for persistent storage
            db_filename = f"graph_state_{student_id}.sqlite" if student_id else "graph_state.sqlite"
            persistence_path = str(graph_path / db_filename)

            # Create a checkpoint instance
            checkpoint = SqliteCheckpoint(persistence_path)

            # Add persistence to the graph
            compiled_graph = compiled_graph.with_state_checkpoint(checkpoint)

        except (ImportError, AttributeError):
            # Fall back to memory-only if SQLite support not available or API mismatch
            pass

    # Return the chain (with persistence if enabled)
    return compiled_graph


# Student-specific conversation utilities
def clear_student_conversation(student_id: str) -> None:
    """Clear conversation history for a specific student."""
    graph_path = get_student_graph_state_path(student_id)
    db_filename = f"graph_state_{student_id}.sqlite"
    db_path = graph_path / db_filename
    
    if db_path.exists():
        db_path.unlink()
        print(f"✅ Cleared conversation history for student: {student_id}")
    else:
        print(f"No conversation history found for student: {student_id}")


def list_student_conversations() -> List[str]:
    """List all students with conversation history."""
    students_dir = settings.graph_state_path.parent / "students"
    if not students_dir.exists():
        return []
    
    students_with_conversations = []
    for student_dir in students_dir.iterdir():
        if student_dir.is_dir():
            graph_state_dir = student_dir / "graph_state"
            if graph_state_dir.exists():
                # Check for SQLite files
                for file in graph_state_dir.glob("graph_state_*.sqlite"):
                    student_id = file.stem.replace("graph_state_", "")
                    if student_id:
                        students_with_conversations.append(student_id)
                        break
    
    return sorted(students_with_conversations)


def migrate_default_conversation_to_student(student_id: str) -> None:
    """Migrate default conversation history to a specific student."""
    default_db = settings.graph_state_path / "graph_state.sqlite"
    
    if not default_db.exists():
        raise FileNotFoundError("No default conversation history to migrate")
    
    student_graph_path = get_student_graph_state_path(student_id)
    student_graph_path.mkdir(parents=True, exist_ok=True)
    
    student_db = student_graph_path / f"graph_state_{student_id}.sqlite"
    
    if student_db.exists():
        raise ValueError(f"Student {student_id} already has conversation history")
    
    import shutil
    shutil.copy2(default_db, student_db)
    print(f"✅ Migrated default conversation history to student: {student_id}")
