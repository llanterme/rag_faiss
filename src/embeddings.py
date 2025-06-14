"""Vector store and embeddings management module."""

from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config import settings


def create_faiss_index(texts: List[str], metadatas: Optional[List[Dict]] = None) -> FAISS:
    """Create a FAISS vector store from a list of text chunks.

    Args:
        texts: List of text chunks to embed.
        metadatas: Optional list of metadata dictionaries for each text chunk.
            Used to store document sources and other information.

    Returns:
        FAISS vector store containing the embedded texts with metadata.
    """
    # Use OpenAI embeddings with API key from settings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=settings.openai_api_key
    )

    # Create FAISS index from texts with metadata
    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return index


def save_index(index: FAISS, path: Optional[Path] = None) -> None:
    """Save a FAISS index to disk.

    Args:
        index: FAISS vector store to save.
        path: Path to save the index to. If None, uses the default path from settings.
    """
    save_path = path or settings.vector_store_path

    # Ensure directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Save the index
    index.save_local(save_path / settings.faiss_index_filename)


def load_index(path: Optional[Path] = None) -> FAISS:
    """Load a FAISS index from disk.

    Args:
        path: Path to load the index from. If None, uses the default path from settings.

    Returns:
        Loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index doesn't exist at the specified path.
    """
    load_path = path or settings.vector_store_path
    full_path = load_path / settings.faiss_index_filename

    if not full_path.exists():
        raise FileNotFoundError(f"No FAISS index found at {full_path}")

    # Use OpenAI embeddings with API key from settings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=settings.openai_api_key
    )

    # Load the index with safe deserialization option
    index = FAISS.load_local(
        full_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return index
