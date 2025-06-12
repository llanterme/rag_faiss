# Document Chat

A Python application that enables users to ingest documents (PDF, DOCX, TXT) and interactively chat or ask questions about them using a modern RAG architecture.

## Features

- **Document Ingestion**: Support for PDF, DOCX, and TXT formats with intelligent chunking
- **Embeddings & Vector Store**: Using OpenAI embeddings and FAISS for local storage
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
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
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
   poetry run python -m src.cli ingest /path/to/document.pdf
   ```

2. Start a chat session:
   ```
   poetry run python -m src.cli chat
   ```

3. View conversation history:
   ```
   poetry run python -m src.cli history
   ```

4. Exit the application:
   ```
   poetry run python -m src.cli exit
   ```

## Requirements

- Python 3.10+
- OpenAI API key

## Tech Stack

- LangChain ^0.3.25
- LangGraph ^0.4.8
- OpenAI ^1.86.0
- FAISS-CPU ^1.11.0
- PyPDF ^5.6.0
- python-docx ^1.1.2
- Typer ^0.16.0
- Streamlit ^1.30.0
- Cryptography ^45.0.4

## Documentation

### Source Attribution

The system now tracks document sources during the retrieval process and displays them alongside answers in the Streamlit UI. This helps users understand which documents were used to generate each response.

### Persistence

Conversation memory is persisted using LangGraph's `SqliteCheckpoint` feature, allowing conversations to be maintained between sessions.

### Custom Formatting

The Streamlit UI provides a cleaner, more visual way to interact with the application, while still maintaining all the functionality of the command-line interface.
