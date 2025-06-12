---
trigger: always_on
---

---
description: Code & Process Standards for Python Projects
globs:
  - "**/*.py"
alwaysApply: true
---

## Always start by reviewing the PLANNING.md file after each prompt to check for feature changes, project overview and features. When a feature is complete, please create or update the TASK.md file to represent the state.

## Never implement features that are outside the scope of your requirements. If you are unsure, please ask for confirmation.

## Every new conversation should automatically begin with Sequential Thinking to determine which other tools are needed for the task at hand.

# Code & Process Standards (Python)

## 1. General Principles

- **Clean Code:**  
  - Follow PEP 8 and The Zen of Python.  
  - Keep functions/classes small (≤ 50 lines). Single responsibility per function/class.  
  - Use descriptive names.  
  - Delete dead code; avoid commented-out code.  
  - Prioritize readability over cleverness.

- **Functional Style & Immutability:**  
  - Favor pure functions without side effects.  
  - Use immutable structures (tuples, `frozenset`) when data shouldn’t change.  
  - Minimize global mutable state.

## 2. Technology Stack

- **Language:** Python 3.10+  
- **Package Management:** Poetry (`pyproject.toml`, `poetry.lock`)  
  - Always use the latest compatible dependencies  
  - Always add all relevant depdedencies to make the profect run.
- **Runtime Environment:**  
  - Always use virtual environments (`poetry shell` or `python -m venv`). No global installs.  
- **AI/ML:**  
  - **LangChain:** ^0.3.25  
  - **OpenAI SDK:** ^1.86.0  
  - **Vector Store:** faiss-cpu ^1.11.0

## 3. MCP Usage 

- **Use context7 MCP server** for up-to-date language documentation.  
- **Use the Sequential Thinking MCP** to break down complex tasks.  
- **Knowledge Graph MCP** should store important findings that might be relevant across conversations.

## 4. Document Ingestion

- Support PDF, DOCX, TXT via `pypdf` ^5.6.0 and `python-docx` ^1.1.2  
- Strip headers/footers; normalize whitespace  
- Split large texts into <1,000-token chunks

## 5. Command-Line Interface

- Use `typer` for commands and auto-generated help  
- Provide commands:  
  - `ingest [path]`  
  - `chat`  
  - `history`  
  - `exit`  
- Validate user inputs; display graceful error messages

## 6. Environment Management

- Support loading environment variables from a `.env` file (e.g., via `python-dotenv` or Pydantic’s built-in dotenv support).  
- Include a `.env.example` template—do **not** commit real secrets.  
- Ensure all sensitive credentials (API keys, tokens) are sourced from environment and never hard-coded.  
- Use the Pydantic `Settings` class to centralize configuration loading.