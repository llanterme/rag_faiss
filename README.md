# Document Chat

A Python application that enables users to ingest documents (PDF, DOCX, TXT) and interactively chat or ask questions about them using a modern RAG architecture.

## Features

- **Document Ingestion**: Support for PDF, DOCX, and TXT formats with intelligent chunking
- **Enhanced Processing**: Advanced extraction with table preservation, metadata extraction, and OCR support
- **Observability**: Complete LLM interaction monitoring with Pydantic Logfire
- **Embeddings & Vector Store**: Using OpenAI or Ollama embeddings and FAISS for local storage
- **LLM Flexibility**: Support for both OpenAI (cloud) and Ollama (local) as LLM backends
- **LangGraph RAG Chain**: Modern graph-based approach with persistent conversation memory
- **Multiple Interfaces**:
  - **Web UI**: Streamlit-based interface with document upload and source attribution
  - **CLI**: Typer-based command-line interface with helpful commands

## Quick Start

### Setup

1. Clone this repository
2. Install dependencies:
   ```
   poetry install
   ```
3. Create a `.env` file with your configuration:
   ```
   # For OpenAI (cloud) provider
   OPENAI_API_KEY=your_api_key_here
   
   # LLM settings (for chat/query)
   # LLM_PROVIDER=openai
   # LLM_MODEL=gpt-4o
   
   # Embedding settings (for document ingestion)
   # EMBEDDING_PROVIDER=openai
   # EMBEDDING_MODEL=text-embedding-ada-002
   
   # For Ollama (local) provider
   # LLM_PROVIDER=ollama
   # LLM_MODEL=llama3.2:latest
   # EMBEDDING_PROVIDER=ollama
   # EMBEDDING_MODEL=llama3.2:latest
   # OLLAMA_BASE_URL=http://localhost:11434
   
   # General settings
   VECTOR_STORE_PATH=./data/vector_store
   ```

### Usage

#### Web UI

Launch the Streamlit web interface:
```
poetry run ui
```
or
```
poetry run streamlit run src/ui.py
```

The web interface allows you to:
- Upload documents directly through the browser
- Chat with your documents in a modern chat interface
- View source attribution for answers (which documents were used)

#### Command Line

1. Ingest documents:
   ```
   poetry run python -m src.cli ingest /path/to/documents
   ```
   
   The system will use the embedding provider and model specified in your configuration.

2. Start a chat session:
   ```
   # Using default LLM provider from configuration
   poetry run python -m src.cli chat
   
   # Override the LLM provider for this session
   poetry run python -m src.cli chat --llm-provider ollama
   
   # Override the LLM model for this session
   poetry run python -m src.cli chat --llm-provider ollama --llm-model llama3.2:latest
   ```

3. View the current configuration:
   ```
   poetry run python -m src.cli provider
   ```
   
   This shows both the LLM configuration (for chat) and embedding configuration (for ingestion).

4. View conversation history:
   ```
   poetry run python -m src.cli history
   ```

5. Exit the application:
   ```dirty
   poetry run python -m src.cli exit
   ```

## Requirements

- Python 3.10+
- One of the following:
  - OpenAI API key (for cloud-based LLM)
  - [Ollama](https://ollama.com/) installed and running locally (for local LLM)

## Tech Stack

- LangChain ^0.3.25
- LangChain-Community ^0.3.25
- LangChain-Ollama ^0.3.3
- LangGraph ^0.4.8
- OpenAI ^1.86.0
- FAISS-CPU ^1.11.0
- PyPDF ^5.6.0
- python-docx ^1.1.2
- Typer ^0.16.0
- Streamlit ^1.30.0
- Cryptography ^45.0.4

## Observability

Monitor your LLM interactions with Pydantic Logfire:

```bash
# Check observability status
poetry run python -m src.cli observability

# View logs in console (default)
poetry run python -m src.cli chat

# Enable cloud dashboard (optional)
echo "LOGFIRE_TOKEN=your_token" >> .env
```

**Features:**
- LLM response times and token usage
- Document processing metrics  
- Embedding creation tracking
- Query performance analysis
- Error tracking and debugging

See [docs/observability.md](docs/observability.md) for complete setup guide.

## Documentation

### Source Attribution

The system now tracks document sources during the retrieval process and displays them alongside answers in the Streamlit UI. This helps users understand which documents were used to generate each response.

### Persistence

Conversation memory is persisted using LangGraph's `SqliteCheckpoint` feature, allowing conversations to be maintained between sessions.

### Custom Formatting

The Streamlit UI provides a cleaner, more visual way to interact with the application, while still maintaining all the functionality of the command-line interface.

### Configuration-Driven Model Selection

The application uses a configuration-driven approach for both LLM and embedding model selection:

#### LLM Configuration (for chat/query)

The application supports two LLM providers:

1. **OpenAI (cloud)**: Requires an API key and internet connection, but offers high-quality responses.
2. **Ollama (local)**: Runs entirely on your local machine for privacy and offline use.

You can configure the LLM provider in several ways:

- Set the `LLM_PROVIDER` and `LLM_MODEL` environment variables in your `.env` file
- Use the `--llm-provider` and `--llm-model` flags with the `chat` command
- Use the `provider` command to view current settings

#### Embedding Configuration (for document ingestion)

Similarly, embeddings can use either OpenAI or Ollama:

- Set the `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` environment variables in your `.env` file
- The embedding configuration can only be changed through the config file, not via CLI arguments

When using Ollama, the default LLM model is `llama3.2:latest` and the default embedding model is also `llama3.2:latest`.

## Known Limitations

- Currently only supports TXT, PDF, and DOCX files for ingestion
- No support for web scraping or direct URL ingestion
- Limited to the capabilities of the underlying LLM model

## Embedding Model Alignment

The system enforces alignment between the embedding provider/model used during ingestion and the one used during querying:

- When documents are ingested, metadata about the embedding model is stored alongside the vector index
- When querying, this metadata is validated against the current configuration
- If there's a mismatch, the system will prevent querying and provide clear instructions to fix the issue
- This prevents silent errors or meaningless retrievals that could occur when using different embedding models

If you change your embedding configuration, you'll need to recreate your vector index by re-ingesting your documents.
