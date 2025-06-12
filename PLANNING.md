# Project Planning

## Overview

We are building a Python-based application that lets users ingest their own documents (PDF, DOCX, TXT, etc.) and interactively chat or ask research‑style questions against that corpus. The core of the system will use LangChain and the OpenAI API for embeddings and chat generation. This file serves as the single source of truth for our coding agent (Windsurf) to understand project scope, architecture, dependencies, conventions, and next steps.

---

## 1. Goals & Features

1. **Document Ingestion**

   * Support common formats: PDF, DOCX, TXT.
   * Extract text, handle basic OCR if needed (future).
   * Clean and normalize text (strip headers/footers).

2. **Embedding & Vector Store**

   * Generate embeddings via `OpenAIEmbeddings`.
   * Use FAISS for local prototyping (Pinecone or Weaviate as future options).
   * Persist index to disk for fast reload.

3. **RetrievalQA Chain**

   * Use LangChain’s `RetrievalQA` with a `ChatOpenAI` model.
   * Implement conversational memory so follow‑ups respect context.

4. **Command-Line Interface**

   * Interactive REPL loop.
   * Commands:  `ingest [path]`, `chat`, `exit`.
   * Simple help menu and clear prompts.

5. **Documentation**

   * Inline docstrings and a `README.md` with usage examples.
   * Keep this `PLANNING.md` up to date.

---

## 2. Architecture & File Structure

```
project_root/
├── PLANNING.md           # This file
├── pyproject.toml        # Poetry config
├── poetry.lock
├── src/
│   ├── __init__.py
│   ├── config.py         # Settings (API keys, paths)
│   ├── ingestion.py      # Document loaders & text splitters
│   ├── embeddings.py     # Embedding setup & vector store
│   ├── qa_chain.py       # RetrievalQA configuration
│   └── cli.py            # CLI REPL interface
```

---

## 3. Dependencies & Versions

* **Python**: 3.10+
* **LangChain**: ^0.0.XXX  (peer‑review latest stable)
* **OpenAI**: ^0.27.5
* **FAISS**: faiss-cpu ^1.7.3
* **PDF Parsing**: `pypdf` ^3.5.1
* **DOCX Parsing**: `python-docx` ^0.8.11
* **CLI**: `typer` ^0.9.0 (optional for advanced CLI features)

Use Poetry to pin these versions in `pyproject.toml`.

---

## 4. Implementation Roadmap

1. **Project Bootstrapping**

   * Initialize with `poetry init`, add dependencies.
   * Create skeleton files per structure above.

2. **Document Ingestion Module** (`ingestion.py`)

   * Implement loaders for `.pdf`, `.docx`, `.txt`.

3. **Vector Store & Embeddings** (`embeddings.py`)

   * Wrap `OpenAIEmbeddings` and FAISS index creation.
   * Support `save_index()` / `load_index()`.

4. **RetrievalQA Chain** (`qa_chain.py`)

   * Configure `RetrievalQA`, integrate memory buffer.


5. **CLI Interface** (`cli.py`)

   * Build REPL loop with Typer or plain `input()`.
   * Wire commands to ingestion and QA chain.
   * Provide clear help and error handling.


6. **Documentation & Release**

   * Update `README.md` with quickstart.
   * Tag v0.1.0 and publish on PyPI (optional).

---

7. **Conventions & Best Practices**

* **Adhere to existing Code & Process Standards** (rules file).
* Log at `INFO` level for user‑visible actions (ingestion, query).
* Use `typer` callbacks for command validation.
* Handle API failures gracefully with retries/backoff.
* Keep all prompts and templates in a dedicated `prompts/` directory if they grow beyond trivial size.

---

*End of PLANNING.md*
