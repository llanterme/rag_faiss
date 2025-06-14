# Migration Task: Add Ollama as a Pluggable LLM Backend

## Context

This project is a Python 3.10+ Retrieval-Augmented Generation (RAG) system. It leverages LangChain, FAISS, Typer CLI, and OpenAI APIs for document ingestion, chunking, vector search, and interactive chat.
All architecture, conventions, and dependency versions are specified in `PLANNING.md` and `pyproject.toml`. The codebase emphasizes maintainability, PEP 8 compliance, and functional design.

* **LLM and embeddings currently use OpenAI (cloud).**
* **All secrets/config handled via Pydantic Settings and dotenv.**
* **Project structure and CLI conventions are defined in `PLANNING.md`.**

---

## Goal

**Enable the project to seamlessly swap between OpenAI (cloud) and Ollama (local) as the LLM backend for chat generation, ensuring privacy, flexibility, and extensibility.**

---

## Tasks

### 1. Add Ollama as an LLM Backend

* Integrate [Ollama](https://ollama.com/) as a local LLM backend, using the [LangChain Ollama integration](https://python.langchain.com/docs/integrations/llms/ollama/).

  * Use `langchain_community.llms.Ollama` or `langchain_community.chat_models.ChatOllama`.
  * Add `langchain-community` to `pyproject.toml` dependencies (ensure version compatibility).
  * Use llama3.2:latest

---

### 2. Abstract LLM Selection

* Refactor LLM initialization logic (in `src/qa_chain.py` or a new module if cleaner) to support runtime selection of LLM provider (`openai` or `ollama`).
* Read the provider from an environment variable (e.g., `LLM_PROVIDER=openai|ollama`) or CLI flag.
* All config should be handled by the existing Pydantic `Settings` class in `src/config.py`.

---

### 3. LLM Initialization & Usage

* Replace or wrap usages of `ChatOpenAI` to transparently support `ChatOllama` based on the selected provider.
* Allow configuration of the Ollama model name via env/CLI (e.g., `LLM_MODEL=llama3`) with a sensible default.
* If using Ollama, **do not require or check for OpenAI API keys**.

---

### 4. CLI Integration

* Update the Typer CLI in `src/cli.py`:

  * Allow users to specify or display the active LLM provider/model.
  * Update CLI help and usage text to document the new local/private LLM option.

---

### 5. Backward Compatibility & Testing

* Ensure current OpenAI workflows continue to function.
* Test both OpenAI and Ollama options for document chat and retrieval.
* Document the provider switch option in `README.md` and update all relevant inline docstrings.

---

### 6. Commit & Document

* Commit all changes with a clear summary (e.g., “Add Ollama local LLM backend and abstract LLM selection”).
* Add a new entry to `TASK.md` describing this migration.
* Update `README.md` with usage examples for both OpenAI and Ollama providers.
* Pause for human review after these changes. **Do not implement further modifications until new instructions are given.**

---

## Project Conventions (Reinforced)

* **Adhere to `PLANNING.md` and workspace coding standards at all times.**
* **Type hints, docstrings, and linting are mandatory.**
* **All configuration is managed via Pydantic and dotenv.**
* **No dependency or feature drift—stick to versions and scope in project docs.**

---

## Summary

> Add Ollama as a swappable LLM backend (using LangChain integration).
> Refactor for runtime provider selection, update CLI/config/docs, ensure backward compatibility.
> Follow all architectural and code conventions, then await further review.

---

*Reference: See [`PLANNING.md`](./PLANNING.md) for detailed project requirements and conventions.*
