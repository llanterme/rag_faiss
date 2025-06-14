"""Command-line interface for the document-chat application."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.embeddings import create_faiss_index, load_index, save_index
from src.ingestion import load_docx, load_pdf, load_txt
from src.langgraph_chain import build_retrieval_chain

app = typer.Typer(help="Document chat application")

# Store the chain and config globally
qa_chain: Any = None
chain_config: Optional[Dict] = None


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to document or directory to ingest")
):
    """Ingest documents into the vector store."""
    if not path.exists():
        typer.echo(f"Error: Path '{path}' does not exist")
        return

    documents: List[Tuple[str, str]] = []  # List of (doc_path, content)

    if path.is_file():
        # Process single file
        if path.suffix.lower() == ".txt":
            typer.echo(f"Ingesting TXT file: {path}")
            documents.append((str(path), load_txt(path)))
        elif path.suffix.lower() == ".pdf":
            typer.echo(f"Ingesting PDF file: {path}")
            documents.append((str(path), load_pdf(path)))
        elif path.suffix.lower() in [".docx", ".doc"]:
            typer.echo(f"Ingesting DOCX file: {path}")
            documents.append((str(path), load_docx(path)))
        else:
            typer.echo(f"Unsupported file type: {path.suffix}")
            return
    elif path.is_dir():
        # Process all supported files in directory
        typer.echo(f"Ingesting files from directory: {path}")
        for file_path in path.glob("**/*"):
            if file_path.is_file():
                if file_path.suffix.lower() == ".txt":
                    typer.echo(f"Ingesting TXT file: {file_path}")
                    documents.append((str(file_path), load_txt(file_path)))
                elif file_path.suffix.lower() == ".pdf":
                    typer.echo(f"Ingesting PDF file: {file_path}")
                    documents.append((str(file_path), load_pdf(file_path)))
                elif file_path.suffix.lower() in [".docx", ".doc"]:
                    typer.echo(f"Ingesting DOCX file: {file_path}")
                    documents.append((str(file_path), load_docx(file_path)))

    if not documents:
        typer.echo("No supported documents found to ingest.")
        return

    # Create text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    # Split documents into chunks
    typer.echo(f"Splitting {len(documents)} document(s) into chunks...")
    chunks = []
    metadatas = []
    for doc_path, content in documents:
        doc_chunks = text_splitter.split_text(content)
        doc_name = Path(doc_path).name
        typer.echo(f"  - {doc_name}: {len(doc_chunks)} chunks")
        
        # Add document source to each chunk's metadata
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadatas.append({"source": doc_name, "source_path": doc_path})

    typer.echo(f"Creating vector embeddings for {len(chunks)} text chunks...")
    index = create_faiss_index(chunks, metadatas=metadatas)

    # Save index to disk
    save_index(index)
    typer.echo(
        f"Index saved to {settings.vector_store_path / settings.faiss_index_filename}"
    )


@app.command()
def chat():
    """Start interactive chat session."""
    global qa_chain, chain_config

    try:
        # Try to load existing index
        index = load_index()
        typer.echo("Loaded existing vector index.")
    except FileNotFoundError:
        typer.echo(
            "Error: No vector index found. Please ingest documents first using 'ingest'."
        )
        return

    # Build retrieval chain with LangGraph
    qa_chain = build_retrieval_chain(index)

    typer.echo("Starting chat session. Type 'exit' to quit.")

    while True:
        # Get user query
        query = typer.prompt("You")

        if query.lower() in ["exit", "quit"]:
            break
            
        # Process the user query through the graph
        try:
            # Use the LangGraph chain with the question
            result = qa_chain.invoke({"question": query})
            
            # LangGraph 0.4.x returns the final state with answer field
            if isinstance(result, dict):
                # First check for direct answer field
                if "answer" in result and result["answer"]:
                    typer.echo(f"AI: {result['answer']}")
                # Then look for messages (especially the last one)
                elif "messages" in result and result["messages"]:
                    for message in result["messages"]:
                        if isinstance(message, AIMessage) and message == result["messages"][-1]:
                            typer.echo(f"AI: {message.content}")
                            break
                else:
                    # Fallback if we can't find the answer
                    typer.echo("AI: I couldn't generate a response. Please try again.")
            else:
                typer.echo(f"AI: {result}")
                
        except Exception as e:
            typer.echo(f"Error processing query: {str(e)}")
            typer.echo("Please try again or restart the chat session.")
            continue


@app.command()
def history():
    """Print conversation history."""
    global qa_chain, chain_config

    if qa_chain is None:
        typer.echo("No active chat session or conversation history.")
        return

    # Get the current state from the LangGraph chain
    try:
        # Different ways to access state depending on LangGraph version
        try:
            # Try direct state access first
            current_state = qa_chain.get_current_state()
        except AttributeError:
            try:
                # Try getting state through config
                if chain_config:
                    current_state = qa_chain.get_state_value(chain_config)
                else:
                    current_state = qa_chain.get_state()
            except Exception:
                # Fall back to direct state if other methods fail
                current_state = qa_chain.get_state() if hasattr(qa_chain, "get_state") else None
        
        # Extract messages from the state
        if current_state and isinstance(current_state, dict) and "messages" in current_state and current_state["messages"]:
            # Print conversation history
            typer.echo("\n--- Conversation History ---")
            for message in current_state["messages"]:
                # Handle different message types
                if isinstance(message, HumanMessage):
                    typer.echo(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    typer.echo(f"AI: {message.content}")
                else:
                    typer.echo(f"Message: {str(message)}")
            typer.echo("---------------------------\n")
        else:
            typer.echo("No messages in conversation history yet. Start a chat to see history.")
    except Exception as e:
        typer.echo(f"Error retrieving conversation history: {str(e)}")
        typer.echo("Try starting a new chat session first.")
        return


@app.command()
def exit():
    """Exit the application."""
    typer.echo("Exiting document-chat application.")
    sys.exit(0)


if __name__ == "__main__":
    # Ensure the vector store directory exists
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    app()
