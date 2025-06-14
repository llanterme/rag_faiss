# Windsurf Project Bootstrap Instruction

You are **Windsurf**, our AI coding agent.
Your goal is to kick off the initial implementation of our **“document-chat”** project by following exactly these instructions:

---

## 1. Read & Honor Our Guides

* Always begin by loading `PLANNING.md` in the repo root.
  Use it as your source of truth for features, file structure, and roadmap.
* Adhere to the `@project-guidelines.md` workspace rules for coding standards, dependency versions, and CLI conventions.

---

## 2. Project Bootstrap

* **Initialize Poetry:**
  Generate `pyproject.toml` targeting Python 3.10+ and pin the exact versions from Section 3 of `PLANNING.md`.
* **Install dependencies** with Poetry.
* **Create the directory structure** under `src/` exactly as specified in `PLANNING.md`.

---

## 3. Implement Skeleton Modules

* **src/config.py**

  * Define a Pydantic `Settings` class reading from environment (`OPENAI_API_KEY`, `VECTOR_STORE_PATH`, etc.).
  * Include default constants for chunk size (e.g., 1,000 tokens) and FAISS index filename.
* **src/ingestion.py**

  * Create three functions with full type hints & docstrings:

    ```python
    def load_txt(path: Path) -> str
    def load_pdf(path: Path) -> str
    def load_docx(path: Path) -> str
    ```
  * Use `pypdf.PdfReader` and `python_docx.Document` to extract text, normalize whitespace, and strip simple headers/footers (placeholders).
* **src/embeddings.py**

  * Stub out functions:

    ```python
    def create_faiss_index(texts: List[str]) -> FAISS
    def save_index(index: FAISS, path: Path) -> None
    def load_index(path: Path) -> FAISS
    ```
  * Import `OpenAIEmbeddings` and `faiss_cpu`.
* **src/qa\_chain.py**

  * Define a function:

    ```python
    def build_retrieval_chain(index: FAISS) -> RetrievalQA
    ```

    that uses `ChatOpenAI` and conversational memory.
* **src/cli.py**

  * Scaffold a Typer app with commands:
    `ingest [path]`, `chat`, `history`, `exit`.
  * `ingest` builds or loads the FAISS index;
    `chat` starts a REPL calling the retrieval chain;
    `history` prints conversation buffer;
    `exit` quits gracefully.

---

## 4. Commit & Task Update

* After scaffolding code, **update `TASK.md`** with a new entry:
  “Bootstrap project structure & skeleton modules.”
* **Do not write any feature code beyond these skeletons yet.**

---

## 5. Follow the Rules

* Use **only** the libraries and versions defined in `PLANNING.md`.

---

**Proceed step-by-step.**
After each file-generation step, confirm by updating `TASK.md`.
Do **not** implement unplanned features or deviate from the specified versions and conventions.
Once the skeleton is in place and passes linting/type checks, **await the next instructions**.

---
