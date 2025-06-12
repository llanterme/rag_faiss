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
        sources: List of document sources for the retrieved context
        answer: Generated AI response
    """

    messages: List[Any]  # HumanMessage or AIMessage
    question: Optional[str]
    context: Optional[str]  # Using string for context, not List[str]
    sources: Optional[List[Dict[str, str]]]  # Document sources with metadata
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
        
        try:
            # Configure retriever with proper settings
            retriever = index.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 documents
            )
            
            # Use invoke method instead of deprecated get_relevant_documents
            docs = retriever.invoke(question)
            
            if not docs:
                print("No documents retrieved for the question")
                return {**state, "context": "", "sources": []}
                
            # Extract document content and source information
            contents = []
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content:
                    contents.append(doc.page_content)
            
            # If no valid content was found
            if not contents:
                print("Retrieved documents had no valid content")
                return {**state, "context": "", "sources": []}
                
            context_str = "\n\n".join(contents)
            
            # Extract source information from metadata
            sources = []
            for doc in docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source_info = {
                        "source": doc.metadata.get("source", "Unknown"),
                        "source_path": doc.metadata.get("source_path", "")
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
            return {**state, "answer": "I couldn't find any relevant information to answer your question."}

        # Create the prompt with conversation history and retrieved context
        messages = [
            SystemMessage(
                content=f"You are a helpful AI assistant answering questions based on the provided context.\n\nContext: {state['context']}"
            ),
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

        # Generate response from LLM
        try:
            print(f"Calling OpenAI API with {len(messages)} messages")
            response = llm.invoke(messages)
            if response and hasattr(response, "content") and response.content:
                return {**state, "answer": response.content}
            else:
                print("Empty or invalid response from LLM")
                return {**state, "answer": "I received an empty response. Please try rephrasing your question."}
        except Exception as e:
            import traceback
            print(f"Answer generation error: {str(e)}")
            print(traceback.format_exc())
            error_msg = f"Error: {str(e)}"
            # For token limit errors, provide a more helpful message
            if "maximum context length" in str(e).lower() or "token" in str(e).lower():
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
        
        # Create new state with updated messages and cleared question/answer/context
        return {
            **state,
            "messages": new_messages,
            "question": None,
            "answer": None,
            "context": None,
            "sources": []
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
        "sources": [],
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
