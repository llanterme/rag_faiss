# Document Chat - Project Analysis

## Overview

Document Chat is a sophisticated Python-based Retrieval-Augmented Generation (RAG) application that allows users to ingest documents in various formats (PDF, DOCX, TXT) and interact with them through natural language conversations. The system uses modern AI techniques including vector embeddings, semantic search, and Large Language Models (LLMs) to provide accurate, contextual answers from uploaded documents.

## Architecture

### Core Components

1. **Document Ingestion Pipeline**
   - Basic ingestion (`src/ingestion.py`): Simple text extraction from PDF, DOCX, and TXT files
   - Enhanced ingestion (`src/enhanced_ingestion.py`): Advanced features including table extraction, OCR support, and formatting preservation
   - Text chunking with configurable size and overlap for optimal retrieval

2. **Vector Storage & Retrieval**
   - FAISS vector database for efficient similarity search
   - Support for both OpenAI and Ollama embeddings
   - Persistent storage with metadata tracking
   - Embedding model alignment enforcement to prevent mismatched queries

3. **LLM Integration**
   - Dual provider support: OpenAI (cloud) and Ollama (local)
   - Configurable models via environment variables or CLI flags
   - LangGraph-based retrieval chain with persistent conversation memory
   - Advanced prompt engineering with multiple style options

4. **User Interfaces**
   - **Streamlit Web UI** (`src/ui.py`): Modern web interface with file upload, chat history, and source attribution
   - **CLI** (`src/cli.py`): Feature-rich command-line interface using Typer

5. **Observability**
   - Pydantic Logfire integration for comprehensive monitoring
   - Tracks LLM response times, token usage, and query performance
   - Error tracking and debugging capabilities

## Key Features

### Document Processing
- **Multiple Format Support**: PDF, DOCX, DOC, and TXT files
- **Enhanced Processing Options**:
  - Table extraction and preservation
  - OCR for scanned documents
  - Formatting preservation
  - Image extraction (optional)
- **Intelligent Chunking**: Recursive text splitting with configurable chunk size and overlap

### Retrieval & Generation
- **Vector Search**: FAISS-based semantic search with top-k retrieval
- **Context-Aware Responses**: LangGraph manages conversation state and memory
- **Source Attribution**: Tracks and displays which documents were used for each answer
- **Prompt Styles**: Multiple prompt templates (default, detailed, concise, academic, technical)

### Configuration & Flexibility
- **Provider Agnostic**: Switch between OpenAI and Ollama for both embeddings and LLM
- **Environment-based Config**: Settings management via `.env` file
- **Runtime Overrides**: CLI flags for dynamic provider/model selection
- **Path Resolution**: Intelligent path handling for consistent operation from any directory

### Persistence & State Management
- **Conversation Memory**: SQLite-based checkpoint system for persistent chat history
- **Vector Index Persistence**: Saved FAISS indices with metadata
- **Graph State Storage**: Maintains conversation context across sessions

## Technical Stack

### Core Dependencies
- **LangChain** (^0.3.25): Foundation for document processing and LLM orchestration
- **LangGraph** (^0.4.8): Modern graph-based approach for conversation management
- **FAISS-CPU** (^1.11.0): Efficient vector similarity search
- **OpenAI** (^1.86.0): Cloud-based LLM and embeddings
- **Ollama** (0.3.3): Local LLM support
- **Streamlit** (^1.30.0): Web UI framework
- **Typer** (^0.16.0): CLI framework

### Document Processing
- **PyPDF** (^5.6.0): PDF text extraction
- **python-docx** (^1.1.2): DOCX processing
- **pdfplumber** (^0.11.0): Enhanced PDF processing
- **PyMuPDF** (^1.24.0): Alternative PDF processing
- **unstructured** (^0.15.0): Advanced document parsing

### Observability & Security
- **Logfire** (^3.21.0): LLM observability
- **Cryptography** (^45.0.4): Secure data handling

## Configuration

The application uses a comprehensive settings system (`src/config.py`) with:
- Provider selection (OpenAI/Ollama) for both LLM and embeddings
- Model configuration with sensible defaults
- Path management with automatic resolution
- Enhanced processing toggles
- Observability settings
- Prompt style selection

## Workflow

1. **Document Ingestion**:
   - User uploads documents via UI or CLI
   - Documents are processed and split into chunks
   - Embeddings are created and stored in FAISS index
   - Metadata is preserved for source tracking

2. **Query Processing**:
   - User asks a question
   - Relevant document chunks are retrieved via semantic search
   - Context is passed to LLM with conversation history
   - Response is generated with source attribution

3. **Conversation Management**:
   - LangGraph maintains conversation state
   - SQLite checkpoint enables persistence
   - Context window is managed automatically

## Security Considerations

- API keys stored in environment variables
- Local option (Ollama) for sensitive data
- No automatic document upload or sharing
- Configurable data persistence

## Performance Optimizations

- Efficient vector indexing with FAISS
- Configurable chunk sizes for optimal retrieval
- Caching mechanisms in web fetch operations
- Parallel processing capabilities

## Extensibility

The modular architecture allows for:
- Adding new document formats
- Implementing custom embedding models
- Creating new UI frontends
- Extending prompt strategies
- Adding new observability providers

## Recent Enhancements

Based on git history, recent improvements include:
- Migration to Ollama support for local LLM deployment
- Enhanced document processing with table/OCR support
- Improved chat functionality
- Better dependency management
- UI/UX improvements in Streamlit interface

## Summary

Document Chat represents a well-architected RAG solution that balances functionality, flexibility, and user experience. Its dual-interface approach (Web UI and CLI) caters to different user preferences, while the provider-agnostic design allows for both cloud and local deployment options. The emphasis on observability and proper error handling makes it suitable for production use cases.