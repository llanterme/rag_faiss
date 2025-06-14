# Task: Ensure Embedding Model and LLM Alignment with Config-Driven Selection

## Background

The current implementation allows flexible LLM and embedding model selection, but **does not enforce alignment** between the embedding provider/model used at ingestion and the one used at query time.
This can lead to silent errors, dimension mismatches, or meaningless retrievals when querying with a different embedding model than was used for ingestion.

Additionally, requiring the LLM/model as a CLI parameter is cumbersome and error-prone.
We want to move to a config-driven approach for setting the target LLM and embedding model.

---

## **Instructions**

### **1. Enforce Embedding Model Alignment**

* When ingesting documents and creating a FAISS (or other vector) index,
  **store metadata** alongside the index indicating:

  * The embedding provider (e.g., `openai`, `ollama`)
  * The embedding model (e.g., `text-embedding-ada-002`, `llama3.2:latest`)
  * The embedding dimension (if available from the model)
  * Date/time of creation (optional, but helpful for audits)
* When loading an index for querying,
  **read and validate this metadata** against the currently configured embedding provider/model.
* If there is a mismatch, **warn the user and prevent querying** until the correct provider/model is selected, or the index is rebuilt.

### **2. Move to Config-Driven Model Selection**

* Replace CLI parameters for LLM and embedding model selection with a configuration file (e.g., `config.yaml` or `.env`).
* The config file should specify:

  * The default LLM provider and model for generation
  * The embedding provider and model for vectorization
  * Any relevant base URLs or API keys
* All ingestion and chat/query flows should read the config file to determine which models/providers to use, **ensuring consistency across the workflow**.

### **3. Implementation Steps**

* Update the ingestion logic to **save metadata** (as JSON or YAML) in the same directory as the index (e.g., `faiss.index.meta.json`).
* Update the index loading/query logic to **read and check** this metadata before proceeding.

  * If a mismatch is detected, show a clear error or prompt to re-ingest.
* Implement or update a config loader that reads the default provider/model from the config file.

  * The CLI should no longer require or accept provider/model parameters.
* Update documentation and help text to explain the new workflow.

---

## **Acceptance Criteria**

* The system **prevents mismatched embedding/querying by default** (no more silent errors or assertion failures).
* All provider/model selection is **config-driven** (no CLI args needed for standard workflows).
* Users are clearly warned and guided if they attempt to query with an incompatible embedding model.
* Documentation is updated to describe this improved, alignment-enforced process.

---

## **References**

* [LangChain vectorstore metadata patterns](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

