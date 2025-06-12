# Document Chat

A Python application that enables users to ingest documents (PDF, DOCX, TXT) and interactively chat or ask questions about them.

## Features

- **Document Ingestion**: Support for PDF, DOCX, and TXT formats
- **Embeddings & Vector Store**: Using OpenAI embeddings and FAISS for local storage
- **QA Chain**: RetrievalQA with ChatOpenAI and conversational memory
- **CLI Interface**: Typer-based interface with helpful commands

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
- OpenAI ^1.86.0
- FAISS-CPU ^1.11.0
- PyPDF ^5.6.0
- python-docx ^1.1.2
- Typer ^0.16.0
