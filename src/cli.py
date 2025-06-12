"""Command-line interface for the document-chat application."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import settings
from src.embeddings import create_faiss_index, load_index, save_index
from src.ingestion import load_docx, load_pdf, load_txt
from src.qa_chain import build_retrieval_chain

app = typer.Typer(help="Document chat application")

# Store the chain globally
qa_chain: Optional[BaseConversationalRetrievalChain] = None


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
    for doc_path, content in documents:
        doc_chunks = text_splitter.split_text(content)
        typer.echo(f"  - {Path(doc_path).name}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)

    typer.echo(f"Creating vector embeddings for {len(chunks)} text chunks...")
    index = create_faiss_index(chunks)

    # Save index to disk
    save_index(index)
    typer.echo(
        f"Index saved to {settings.vector_store_path / settings.faiss_index_filename}"
    )


@app.command()
def chat():
    """Start interactive chat session."""
    global qa_chain

    try:
        # Try to load existing index
        index = load_index()
        typer.echo("Loaded existing vector index.")
    except FileNotFoundError:
        typer.echo(
            "Error: No vector index found. Please ingest documents first using 'ingest'."
        )
        return

    # Build retrieval chain
    qa_chain = build_retrieval_chain(index)

    typer.echo("Starting chat session. Type 'exit' to quit.")

    while True:
        # Get user query
        query = typer.prompt("You")

        if query.lower() in ["exit", "quit"]:
            break

        # Get response
        response = qa_chain({"question": query})
        typer.echo(f"AI: {response['answer']}")


@app.command()
def history():
    """Print conversation history."""
    global qa_chain

    if qa_chain is None or qa_chain.memory is None:
        typer.echo("No active chat session or conversation history.")
        return

    # Print conversation history from memory
    for message in qa_chain.memory.chat_memory.messages:
        typer.echo(f"{message.type.capitalize()}: {message.content}")


@app.command()
def exit():
    """Exit the application."""
    typer.echo("Exiting document-chat application.")
    sys.exit(0)


if __name__ == "__main__":
    # Ensure the vector store directory exists
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    app()
