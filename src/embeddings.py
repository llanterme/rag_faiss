"""Vector store and embeddings management module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import LLMProvider, EmbeddingProvider, settings


def get_embeddings() -> Embeddings:
    """Get the appropriate embeddings model based on settings.
    
    Returns:
        Embeddings model instance based on the configured provider.
    """
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(
            model=settings.embedding_model, 
            openai_api_key=settings.openai_api_key
        )
    elif settings.embedding_provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def get_embedding_dimension(embeddings: Embeddings) -> int:
    """Get the dimension of the embeddings model.
    
    Args:
        embeddings: The embeddings model to get the dimension for.
        
    Returns:
        The dimension of the embeddings model.
    """
    # Create a simple embedding to determine the dimension
    try:
        # Use a simple text to get embedding dimension
        test_embedding = embeddings.embed_query("test")
        return len(test_embedding)
    except Exception as e:
        # If we can't determine dimension, return None
        print(f"Warning: Could not determine embedding dimension: {str(e)}")
        return 0


def create_embedding_metadata() -> Dict[str, Any]:
    """Create metadata for the current embedding configuration.
    
    Returns:
        Dictionary containing metadata about the embedding model.
    """
    embeddings = get_embeddings()
    dimension = get_embedding_dimension(embeddings)
    
    metadata = {
        "embedding_provider": settings.embedding_provider.value,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": dimension,
        "created_at": datetime.now().isoformat(),
    }
    
    return metadata


def save_embedding_metadata(path: Optional[Path] = None) -> None:
    """Save embedding metadata to disk.
    
    Args:
        path: Path to save the metadata to. If None, uses the default path from settings.
    """
    save_path = path or settings.vector_store_path
    metadata_path = save_path / settings.faiss_metadata_filename
    
    # Create metadata
    metadata = create_embedding_metadata()
    
    # Save metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_embedding_metadata(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load embedding metadata from disk.
    
    Args:
        path: Path to load the metadata from. If None, uses the default path from settings.
        
    Returns:
        Dictionary containing metadata about the embedding model.
        
    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
    """
    load_path = path or settings.vector_store_path
    metadata_path = load_path / settings.faiss_metadata_filename
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata file found at {metadata_path}")
    
    # Load metadata from JSON
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def validate_embedding_compatibility(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that the current embedding configuration is compatible with the metadata.
    
    Args:
        metadata: Dictionary containing metadata about the embedding model.
        
    Returns:
        Tuple of (is_compatible, message) where is_compatible is a boolean indicating
        whether the current configuration is compatible with the metadata, and message
        is a string explaining why if not compatible.
    """
    # Check provider
    if metadata.get("embedding_provider") != settings.embedding_provider.value:
        return False, f"Embedding provider mismatch: index was created with {metadata.get('embedding_provider')} but current provider is {settings.embedding_provider.value}"
    
    # Check model
    if metadata.get("embedding_model") != settings.embedding_model:
        return False, f"Embedding model mismatch: index was created with {metadata.get('embedding_model')} but current model is {settings.embedding_model}"
    
    return True, "Compatible"


def create_faiss_index(texts: List[str], metadatas: Optional[List[Dict]] = None) -> FAISS:
    """Create a FAISS vector store from a list of text chunks.

    Args:
        texts: List of text chunks to embed.
        metadatas: Optional list of metadata dictionaries for each text chunk.
            Used to store document sources and other information.

    Returns:
        FAISS vector store containing the embedded texts with metadata.
    """
    # Get embeddings based on the configured provider
    embeddings = get_embeddings()

    # Create FAISS index from texts with metadata
    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return index


def save_index(index: FAISS, path: Optional[Path] = None) -> None:
    """Save a FAISS index to disk along with metadata about the embedding model.

    Args:
        index: FAISS vector store to save.
        path: Path to save the index to. If None, uses the default path from settings.
    """
    save_path = path or settings.vector_store_path

    # Ensure directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Save the index
    index.save_local(save_path / settings.faiss_index_filename)
    
    # Save metadata about the embedding model
    save_embedding_metadata(save_path)


def load_index(path: Optional[Path] = None) -> FAISS:
    """Load a FAISS index from disk and validate embedding compatibility.

    Args:
        path: Path to load the index from. If None, uses the default path from settings.

    Returns:
        Loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index doesn't exist at the specified path.
        ValueError: If there's a dimension mismatch or incompatible embedding model.
    """
    load_path = path or settings.vector_store_path
    full_path = load_path / settings.faiss_index_filename

    if not full_path.exists():
        raise FileNotFoundError(f"No FAISS index found at {full_path}")
    
    # Check for metadata file and validate compatibility
    try:
        metadata = load_embedding_metadata(load_path)
        is_compatible, message = validate_embedding_compatibility(metadata)
        
        if not is_compatible:
            print(f"\nERROR: {message}")
            print("\nTo fix this, you have two options:")
            print("\n1. Use the same embedding provider and model that created the index:")
            print(f"   Provider: {metadata.get('embedding_provider')}")
            print(f"   Model: {metadata.get('embedding_model')}")
            print("\n   Update your .env file or settings to match these values.")
            
            print("\n2. Recreate the index with your current embedding configuration:")
            print(f"   rm -rf {full_path} {load_path / settings.faiss_metadata_filename}")
            print(f"   poetry run python -m src.cli ingest /path/to/your/documents")
            
            raise ValueError(f"Incompatible embedding configuration. {message}")
            
    except FileNotFoundError:
        # No metadata file found, this is likely an older index
        print("\nWARNING: No metadata file found for this index.")
        print("This index was created with an older version of the application.")
        print("We'll attempt to load it with the current embedding configuration,")
        print("but if this fails, you'll need to recreate the index.")

    # Try to load the index with the current provider's embeddings
    current_embeddings = get_embeddings()
    
    try:
        # Try to load the index with the current provider's embeddings
        index = FAISS.load_local(
            full_path, 
            current_embeddings,
            allow_dangerous_deserialization=True
        )
        return index
    except AssertionError as e:
        # If we get here, there's a dimension mismatch
        print("\nERROR: Embedding dimension mismatch between your index and current provider.")
        print(f"Your index at {full_path} was created with a different embedding model.")
        
        if 'metadata' in locals():
            print(f"\nIndex was created with:")
            print(f"  - Provider: {metadata.get('embedding_provider')}")
            print(f"  - Model: {metadata.get('embedding_model')}")
            print(f"  - Dimension: {metadata.get('embedding_dimension')}")
        
        print("\nCurrent configuration:")
        print(f"  - Provider: {settings.embedding_provider.value}")
        print(f"  - Model: {settings.embedding_model}")
        
        print("\nTo fix this, you have two options:")
        print("\n1. Update your configuration to match the index:")
        if 'metadata' in locals():
            print(f"   - Set embedding_provider={metadata.get('embedding_provider')}")
            print(f"   - Set embedding_model={metadata.get('embedding_model')}")
        
        print("\n2. Recreate the index with your current configuration:")
        print(f"   rm -rf {full_path} {load_path / settings.faiss_metadata_filename}")
        print(f"   poetry run python -m src.cli ingest /path/to/your/documents")
        
        raise ValueError("Embedding dimension mismatch. Please recreate the index or update your configuration.")
