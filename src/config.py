"""Configuration settings for the document chat application."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the document chat application.

    Attributes:
        openai_api_key (str): OpenAI API key for embeddings and chat completion.
        vector_store_path (Path): Path to store the FAISS vector index.
        chunk_size (int): Size of document chunks for embedding.
        chunk_overlap (int): Overlap between document chunks.
        faiss_index_filename (str): Filename for the FAISS index.
        temperature (float): Temperature for the OpenAI model.
    """

    openai_api_key: str = Field(
        default="", description="OpenAI API key for embeddings and chat completion"
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
    temperature: float = Field(
        default=0.0, description="Temperature for the OpenAI model"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a singleton instance of settings
settings = Settings()
