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

try:
    from src.observability import logfire_manager, log_operation

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

    # Create no-op decorators if observability is not available
    def log_operation(name):
        def decorator(func):
            return func

        return decorator

    class MockLogfireManager:
        def span(self, *args, **kwargs):
            from contextlib import nullcontext

            return nullcontext()

        def log_embedding_creation(self, *args, **kwargs):
            pass

    logfire_manager = MockLogfireManager()


def get_embeddings() -> Embeddings:
    """Get the appropriate embeddings model based on settings.

    Returns:
        Embeddings model instance based on the configured provider.
    """
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    elif settings.embedding_provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider}"
        )


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


def get_student_index_path(student_id: Optional[str] = None) -> Path:
    """Resolve index path for student or default location."""
    if student_id:
        return (
            settings.vector_store_path.parent / "students" / student_id / "vector_store"
        )
    return settings.vector_store_path


def _add_student_metadata(metadatas: List[Dict], student_id: str) -> None:
    """Add student identifier to metadata entries."""
    for metadata in metadatas:
        metadata["student_id"] = student_id


def _create_openai_index(
    texts: List[str], embeddings: Embeddings, metadatas: Optional[List[Dict]]
) -> FAISS:
    """Create FAISS index using OpenAI with token batching."""
    return _create_faiss_index_with_token_batching(texts, embeddings, metadatas)


def _create_ollama_index(
    texts: List[str], embeddings: Embeddings, metadatas: Optional[List[Dict]]
) -> FAISS:
    """Create FAISS index using Ollama with simple batching."""
    batch_size = 5000
    if len(texts) <= batch_size:
        return FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return _create_batched_index(texts, embeddings, metadatas, batch_size)


def _create_batched_index(
    texts: List[str],
    embeddings: Embeddings,
    metadatas: Optional[List[Dict]],
    batch_size: int,
) -> FAISS:
    """Create FAISS index using batch processing."""
    print(f"Processing {len(texts)} chunks in batches of {batch_size}...")

    first_batch_texts = texts[:batch_size]
    first_batch_metadatas = metadatas[:batch_size] if metadatas else None
    index = FAISS.from_texts(
        first_batch_texts, embeddings, metadatas=first_batch_metadatas
    )

    for i in range(batch_size, len(texts), batch_size):
        batch_end = min(i + batch_size, len(texts))
        batch_texts = texts[i:batch_end]
        batch_metadatas = metadatas[i:batch_end] if metadatas else None

        print(f"Processing batch {i//batch_size + 1}: chunks {i+1}-{batch_end}")

        batch_index = FAISS.from_texts(
            batch_texts, embeddings, metadatas=batch_metadatas
        )
        index.merge_from(batch_index)

    return index


def create_embedding_metadata(student_id: Optional[str] = None) -> Dict[str, Any]:
    """Create metadata for current embedding configuration."""
    embeddings = get_embeddings()
    dimension = get_embedding_dimension(embeddings)

    metadata = {
        "embedding_provider": settings.embedding_provider.value,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": dimension,
        "created_at": datetime.now().isoformat(),
    }

    if student_id:
        metadata["student_id"] = student_id

    return metadata


def save_embedding_metadata(
    student_id: Optional[str] = None, path: Optional[Path] = None
) -> None:
    """Save embedding metadata to disk."""
    save_path = path or get_student_index_path(student_id)
    metadata_path = save_path / settings.faiss_metadata_filename

    metadata = create_embedding_metadata(student_id)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_embedding_metadata(
    student_id: Optional[str] = None, path: Optional[Path] = None
) -> Dict[str, Any]:
    """Load embedding metadata from disk."""
    load_path = path or get_student_index_path(student_id)
    metadata_path = load_path / settings.faiss_metadata_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata file found at {metadata_path}")

    with open(metadata_path, "r") as f:
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
        return (
            False,
            f"Embedding provider mismatch: index was created with {metadata.get('embedding_provider')} but current provider is {settings.embedding_provider.value}",
        )

    # Check model
    if metadata.get("embedding_model") != settings.embedding_model:
        return (
            False,
            f"Embedding model mismatch: index was created with {metadata.get('embedding_model')} but current model is {settings.embedding_model}",
        )

    return True, "Compatible"


@log_operation("create_faiss_index")
def create_faiss_index(
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    student_id: Optional[str] = None,
) -> FAISS:
    """Create FAISS vector store from text chunks.

    Args:
        texts: Text chunks to embed.
        metadatas: Metadata for each chunk.
        student_id: Optional student identifier for isolation.

    Returns:
        FAISS vector store with embedded texts.
    """
    import time

    start_time = time.time()

    with logfire_manager.span("embedding_creation") as span:
        span.set_attribute("chunk_count", len(texts))
        span.set_attribute("total_text_length", sum(len(text) for text in texts))
        span.set_attribute("embedding_provider", str(settings.embedding_provider))
        span.set_attribute("embedding_model", settings.embedding_model)
        span.set_attribute("has_metadata", metadatas is not None)

        embeddings = get_embeddings()

        if student_id and metadatas:
            _add_student_metadata(metadatas, student_id)

        if settings.embedding_provider == EmbeddingProvider.OPENAI:
            index = _create_openai_index(texts, embeddings, metadatas)
        else:
            index = _create_ollama_index(texts, embeddings, metadatas)

        processing_time = time.time() - start_time
        span.set_attribute("processing_time_seconds", processing_time)

        # Log embedding creation metrics
        logfire_manager.log_embedding_creation(
            chunk_count=len(texts),
            processing_time=processing_time,
            provider=str(settings.embedding_provider),
        )

    return index


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    This is a rough approximation: 1 token ≈ 4 characters for English text.
    OpenAI's actual tokenizer is more complex, but this provides a safe estimate.
    """
    return len(text) // 3  # Conservative estimate: 3 chars per token


def _create_faiss_index_with_token_batching(
    texts: List[str], embeddings: Embeddings, metadatas: Optional[List[dict]] = None
) -> FAISS:
    """Create FAISS index with token-aware batching for OpenAI's limits.

    OpenAI has a 300k token limit per request. This function batches texts
    to stay safely under that limit.
    """
    MAX_TOKENS_PER_BATCH = 250000  # Leave some safety margin

    batches: List[Tuple[List[str], List[Dict]]] = []
    current_batch_texts: List[str] = []
    current_batch_metadatas: List[Dict] = []
    current_batch_tokens = 0

    print(f"Organizing {len(texts)} chunks into token-aware batches...")

    for i, text in enumerate(texts):
        text_tokens = _estimate_tokens(text)

        # If adding this text would exceed the limit, start a new batch
        if (
            current_batch_tokens + text_tokens > MAX_TOKENS_PER_BATCH
            and current_batch_texts
        ):
            batches.append((current_batch_texts.copy(), current_batch_metadatas.copy()))
            current_batch_texts = []
            current_batch_metadatas = []
            current_batch_tokens = 0

        current_batch_texts.append(text)
        if metadatas:
            current_batch_metadatas.append(metadatas[i])
        current_batch_tokens += text_tokens

    # Add the last batch if it has content
    if current_batch_texts:
        batches.append((current_batch_texts, current_batch_metadatas))

    print(f"Created {len(batches)} batches with estimated token counts:")
    for i, (batch_texts, _) in enumerate(batches):
        batch_tokens = sum(_estimate_tokens(text) for text in batch_texts)
        print(f"  Batch {i+1}: {len(batch_texts)} chunks, ~{batch_tokens:,} tokens")

    # Process the first batch to create the initial index
    first_batch_texts, first_batch_metadatas = batches[0]
    print(f"Processing batch 1/{len(batches)}...")
    index = FAISS.from_texts(
        first_batch_texts,
        embeddings,
        metadatas=first_batch_metadatas if metadatas else None,
    )

    # Process remaining batches and merge them
    for i, (batch_texts, batch_metadatas) in enumerate(batches[1:], 2):
        print(f"Processing batch {i}/{len(batches)}...")
        batch_index = FAISS.from_texts(
            batch_texts, embeddings, metadatas=batch_metadatas if metadatas else None
        )
        index.merge_from(batch_index)

    print(f"✅ Successfully created index with {index.index.ntotal} vectors")
    return index


def save_index(
    index: FAISS, student_id: Optional[str] = None, path: Optional[Path] = None
) -> None:
    """Save FAISS index to disk with embedding metadata."""
    save_path = path or get_student_index_path(student_id)
    save_path.mkdir(parents=True, exist_ok=True)

    index.save_local(str(save_path / settings.faiss_index_filename))
    save_embedding_metadata(student_id, save_path)
    
    if student_id:
        print(f"✅ Saved index for student: {student_id}")
        _validate_student_index_integrity(student_id)


def load_index(student_id: Optional[str] = None, path: Optional[Path] = None) -> FAISS:
    """Load FAISS index from disk and validate compatibility."""
    load_path = path or get_student_index_path(student_id)
    full_path = load_path / settings.faiss_index_filename

    if not full_path.exists():
        if student_id:
            raise FileNotFoundError(
                f"No FAISS index found for student '{student_id}' at {full_path}. "
                f"Available students: {list_student_indexes()}"
            )
        raise FileNotFoundError(f"No FAISS index found at {full_path}")

    try:
        metadata = load_embedding_metadata(student_id, load_path)
        is_compatible, message = validate_embedding_compatibility(metadata)

        if not is_compatible:
            print(f"\nERROR: {message}")
            print("\nTo fix this, you have two options:")
            print(
                "\n1. Use the same embedding provider and model that created the index:"
            )
            print(f"   Provider: {metadata.get('embedding_provider')}")
            print(f"   Model: {metadata.get('embedding_model')}")
            print("\n   Update your .env file or settings to match these values.")

            print("\n2. Recreate the index with your current embedding configuration:")
            print(
                f"   rm -rf {full_path} {load_path / settings.faiss_metadata_filename}"
            )
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
            str(full_path), current_embeddings, allow_dangerous_deserialization=True
        )
        return index
    except AssertionError as e:
        # If we get here, there's a dimension mismatch
        print(
            "\nERROR: Embedding dimension mismatch between your index and current provider."
        )
        print(
            f"Your index at {full_path} was created with a different embedding model."
        )

        if "metadata" in locals():
            print(f"\nIndex was created with:")
            print(f"  - Provider: {metadata.get('embedding_provider')}")
            print(f"  - Model: {metadata.get('embedding_model')}")
            print(f"  - Dimension: {metadata.get('embedding_dimension')}")

        print("\nCurrent configuration:")
        print(f"  - Provider: {settings.embedding_provider.value}")
        print(f"  - Model: {settings.embedding_model}")

        print("\nTo fix this, you have two options:")
        print("\n1. Update your configuration to match the index:")
        if "metadata" in locals():
            print(f"   - Set embedding_provider={metadata.get('embedding_provider')}")
            print(f"   - Set embedding_model={metadata.get('embedding_model')}")

        print("\n2. Recreate the index with your current configuration:")
        print(f"   rm -rf {full_path} {load_path / settings.faiss_metadata_filename}")
        print(f"   poetry run python -m src.cli ingest /path/to/your/documents")

        raise ValueError(
            "Embedding dimension mismatch. Please recreate the index or update your configuration."
        )


# Student-specific convenience functions
def create_student_faiss_index(
    texts: List[str], student_id: str, metadatas: Optional[List[Dict]] = None
) -> FAISS:
    """Create FAISS index for specific student."""
    return create_faiss_index(texts, metadatas, student_id)


def save_student_index(index: FAISS, student_id: str) -> None:
    """Save FAISS index for specific student."""
    save_index(index, student_id)


def load_student_index(student_id: str) -> FAISS:
    """Load FAISS index for specific student."""
    return load_index(student_id)


def student_index_exists(student_id: str) -> bool:
    """Check if FAISS index exists for student."""
    index_path = get_student_index_path(student_id) / settings.faiss_index_filename
    return index_path.exists()


def list_student_indexes() -> List[str]:
    """List all available student indexes."""
    students_dir = settings.vector_store_path.parent / "students"
    if not students_dir.exists():
        return []

    student_ids = []
    for student_dir in students_dir.iterdir():
        if student_dir.is_dir() and student_index_exists(student_dir.name):
            student_ids.append(student_dir.name)

    return sorted(student_ids)


def _validate_student_index_integrity(student_id: str) -> bool:
    """Validate student index has all required files."""
    base_path = get_student_index_path(student_id)
    index_path = base_path / settings.faiss_index_filename
    metadata_path = base_path / settings.faiss_metadata_filename
    
    # Check index files
    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(f"Missing index.faiss for student {student_id}")
    if not (index_path / "index.pkl").exists():
        raise FileNotFoundError(f"Missing index.pkl for student {student_id}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for student {student_id}")
    
    # Validate metadata contains student_id
    try:
        metadata = load_embedding_metadata(student_id)
        if metadata.get("student_id") != student_id:
            raise ValueError(
                f"Metadata student_id mismatch: expected '{student_id}', "
                f"found '{metadata.get('student_id')}'"
            )
    except Exception as e:
        raise ValueError(f"Invalid metadata for student {student_id}: {e}")
    
    return True


def get_student_document_count(student_id: str) -> int:
    """Get the number of documents in a student's index."""
    if not student_index_exists(student_id):
        return 0
    
    try:
        index = load_student_index(student_id)
        return index.index.ntotal
    except Exception:
        return 0


def migrate_default_to_student(student_id: str, backup: bool = True) -> None:
    """Migrate default index to a specific student."""
    default_index_path = settings.vector_store_path / settings.faiss_index_filename
    
    if not default_index_path.exists():
        raise FileNotFoundError("No default index to migrate")
    
    if student_index_exists(student_id):
        raise ValueError(f"Student {student_id} already has an index")
    
    # Load default index
    default_index = load_index()
    
    # Save as student index
    save_student_index(default_index, student_id)
    
    if backup:
        # Create backup of default index
        import shutil
        backup_path = settings.vector_store_path.parent / "backup_default_index"
        shutil.copytree(settings.vector_store_path, backup_path)
        print(f"✅ Backed up default index to {backup_path}")
    
    print(f"✅ Migrated default index to student: {student_id}")
