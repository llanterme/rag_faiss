"""Configuration settings for the document chat application."""

import os
import json
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Enum for supported embedding providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Settings for the document chat application.

    Attributes:
        llm_provider (LLMProvider): LLM provider to use (openai or ollama).
        llm_model (str): Model name to use with the selected provider.
        embedding_provider (EmbeddingProvider): Embedding provider to use (openai or ollama).
        embedding_model (str): Embedding model name to use with the selected provider.
        openai_api_key (str): OpenAI API key for embeddings and chat completion.
        ollama_base_url (str): Base URL for Ollama API.
        vector_store_path (Path): Path to store the FAISS vector index.
        chunk_size (int): Size of document chunks for embedding.
        chunk_overlap (int): Overlap between document chunks.
        faiss_index_filename (str): Filename for the FAISS index.
        faiss_metadata_filename (str): Filename for the FAISS index metadata.
        temperature (float): Temperature for the LLM model.
        graph_state_path (Path): Path to store persistent graph state data.
        enable_persistence (bool): Whether to enable graph state persistence.
    """

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use (openai or ollama)"
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name to use with the selected provider"
    )
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Embedding provider to use (openai or ollama)"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model name to use with the selected provider"
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key for embeddings and chat completion"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    vector_store_path: Path = Field(
        default=Path("./data/vector_store"),
        description="Path to store the FAISS vector index",
    )
    chunk_size: int = Field(
        default=1000, description="Size of document chunks for embedding"
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between document chunks"
    )
    faiss_index_filename: str = Field(
        default="faiss_index", description="Filename for the FAISS index"
    )
    faiss_metadata_filename: str = Field(
        default="faiss_index.meta.json", description="Filename for the FAISS index metadata"
    )
    temperature: float = Field(
        default=0.0, description="Temperature for the OpenAI model"
    )
    graph_state_path: Path = Field(
        default=Path("./data/graph_state"),
        description="Path to store persistent graph state data"
    )
    enable_persistence: bool = Field(
        default=True, description="Whether to enable graph state persistence"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        # Make sure paths are absolute by resolving them relative to the project root
        self._fix_paths()
    
    def _fix_paths(self) -> None:
        """Ensure all paths are absolute regardless of how the app is executed."""
        # Determine the project root directory by locating the pyproject.toml
        # This works whether we're running from the project root or any subdirectory
        current_dir = Path().absolute()
        project_root = self._find_project_root(current_dir)
        
        # Resolve relative paths to absolute paths
        if not self.vector_store_path.is_absolute():
            # If the path starts with ./ or ../, resolve from project root
            if str(self.vector_store_path).startswith("./") or str(self.vector_store_path).startswith("../"):
                self.vector_store_path = project_root / self.vector_store_path.relative_to(Path("./"))
            else:
                self.vector_store_path = project_root / self.vector_store_path
            
        if not self.graph_state_path.is_absolute():
            if str(self.graph_state_path).startswith("./") or str(self.graph_state_path).startswith("../"):
                self.graph_state_path = project_root / self.graph_state_path.relative_to(Path("./"))
            else:
                self.graph_state_path = project_root / self.graph_state_path
    
    def _find_project_root(self, start_dir: Path) -> Path:
        """Find the project root by looking for pyproject.toml"""
        current = start_dir
        # Traverse up to 5 parent directories looking for pyproject.toml
        for _ in range(5):
            if (current / "pyproject.toml").exists():
                return current
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        
        # Fallback to the current directory if project root not found
        return start_dir


# Create a singleton instance of settings
settings = Settings()
