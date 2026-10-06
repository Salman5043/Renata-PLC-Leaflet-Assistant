# Renata PLC Leaflet Assistant

A document-grounded **Retrieval-Augmented Generation (RAG)** assistant for answering questions from the supplied Renata PLC medicine leaflets.

The assistant retrieves relevant information from the supplied PDF leaflets, generates an answer using a Groq-hosted LLM, and provides supporting document citations. The system is designed to answer only from the supplied leaflet content and abstain when sufficient evidence is not available.

---

## 1. Requirements

Before running the project, make sure the following are installed:

* Python 3.12+
* `uv` or `pip`
* Groq API key

The project can be run using either `uv` or standard Python `pip`.

---

# 2. Project Structure

The important project directories are:

```text
Renata-PLC-Leaflet-Assistant/
│
├── app/
│   ├── main.py
│   └── ...
│
├── scripts/
│   └── ingest.py
│
├── evaluation/
│   └── evaluate.py
│
├── tests/
│   └── test_core.py
│
├── data/
│   └── Renata PLC leaflet PDFs
│
├── chroma_db/
│   └── vector database
│
├── .env
├── pyproject.toml
├── requirements.txt
└── README.md
```

Run all commands from the **project root directory**.

---


# 4. Option A — Setup Using `uv`

`uv` is the recommended way to install and run the project.

## 4.1 Install Dependencies

From the project root:

```powershell
uv sync
```

This creates or uses the project's virtual environment and installs the dependencies defined in `pyproject.toml`.

Verify Python:

```powershell
uv run python --version
```

---

# 5. Option B — Setup Using `pip`

If `uv` is not installed, use standard Python.

## 5.1 Create Virtual Environment

```powershell
python -m venv .venv
```

## 5.2 Activate Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, using Command Prompt:

```cmd
.venv\Scripts\activate
```

## 5.3 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 6. Build the Vector Database

Before starting the application, the supplied Renata PLC leaflet PDFs must be indexed.

The ingestion process:

1. Loads the supplied PDF leaflets.
2. Extracts their text.
3. Splits the content into retrieval chunks.
4. Generates embeddings.
5. Stores the embeddings and metadata in ChromaDB.

## Using `uv`

```powershell
uv run python -m scripts.ingest
```

## Using `pip`

```powershell
python -m scripts.ingest
```

A successful ingestion should produce output similar to:

```text
Indexed PDFs: 5
Section/page blocks: ...
Vector chunks: ...
```

The exact number of blocks and chunks depends on the supplied documents and chunking configuration.

---

# 7. Start the Application

## Using `uv`

```powershell
uv run uvicorn app.main:app --reload
```

## Using `pip`

```powershell
uvicorn app.main:app --reload
```

The FastAPI application will start at:

```text
http://127.0.0.1:8000
```

---

# 8. Open the Chat UI

After starting the application, open the following address in a web browser:

```text
http://127.0.0.1:8000
```

The browser will display the **Renata PLC Leaflet Assistant** chat interface.

You can then ask questions such as:

```text
What is Rolac and what is it used for?
```

or:

```text
What is the usual adult dose of Rolip?
```

The application returns the generated answer together with supporting leaflet citations.

---

# 9. Application Flow

The overall RAG pipeline is:

```text
User Question
      |
      v
Product / Intent Detection
      |
      v
ChromaDB Retrieval
      |
      v
Hybrid Ranking
(Dense similarity + lexical overlap + section relevance)
      |
      v
Retrieved Leaflet Evidence
      |
      v
Groq LLM
      |
      v
Citation Validation
      |
      v
Grounding Validation
      |
      +------ Unsupported ------> Abstain
      |
      v
Final Answer + Citations
```

The system is designed to remain grounded in the supplied Renata PLC leaflet documents.

---

# 10. Retrieval Strategy

The assistant uses a hybrid retrieval/ranking approach rather than relying only on vector similarity.

The ranking considers:

* Dense semantic similarity
* Lexical overlap
* Section relevance
* Product relevance

This helps improve retrieval for both product-specific and general questions.

For example:

### Product-specific question

```text
What is Rolac and what is it used for?
```

The system prioritizes evidence associated with the Rolac leaflet.

### General question

```text
Use in pregnancy and breast-feeding
```

The system can retrieve relevant evidence from multiple supplied leaflets instead of unnecessarily limiting the answer to one document.

---

# 11. LLM Model

The application uses **Groq via LangChain's `ChatGroq`**.

The LLM is used for:

1. Generating the final answer from retrieved leaflet evidence.
2. Performing a second-pass grounding validation.

The exact model is controlled by the environment variable:

```env
GROQ_MODEL=your_groq_model
```

For example, if the project configuration uses a supported Groq model, that model will be loaded through `ChatGroq`.

The model is configured with a low/zero temperature to make responses more deterministic and grounded.

---

# 12. Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts both:

* Leaflet document chunks
* User questions

into vector representations.

These vectors are stored in ChromaDB and used for semantic similarity retrieval.

---

# 13. Vector Database

The project uses:

```text
ChromaDB
```

ChromaDB stores:

* Document embeddings
* Retrieved text chunks
* Document metadata
* Source/page information used for citations

The vector database is generated during the ingestion process:

```powershell
uv run python scripts/ingest.py
```

---

# 14. Running the Tests

The project contains a core test module:

```text
tests/test_core.py
```

## Run the Core Test Suite

Using Python:

```powershell
python -m tests.test_core
```

Using `uv`:

```powershell
uv run python -m tests.test_core
```

---

# 15. Running the Evaluation

The project contains an evaluation module:

```text
evaluation/evaluate.py
```

Run it from the project root.

## Using Python

```powershell
python -m evaluation.evaluate
```

## Using `uv`

```powershell
uv run python -m evaluation.evaluate
```

The evaluation script tests the RAG pipeline using predefined questions and evaluates the system's retrieval and answer behavior.

---

# 16. Running Pytest

The project can also be tested using `pytest`.

Run the complete test suite:

```powershell
pytest -v
```

Or with `uv`:

```powershell
uv run pytest -v
```

---

# 17. Recommended Validation Sequence

After making changes to the ingestion, retrieval, ranking, or generation pipeline, run:

```powershell
uv run python -m scripts.ingest
```

Then:

```powershell
uv run python -m tests.test_core
```

Then:

```powershell
uv run python -m evaluation.evaluate
```

Finally start the application:

```powershell
uv run uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

This provides the complete validation workflow:

```text
PDF Documents
      |
      v
Ingestion
      |
      v
ChromaDB
      |
      v
Core Tests
      |
      v
Evaluation
      |
      v
Chat Application
```

---

# 18. Quick Start

If `uv` is already installed:

```powershell
uv sync

uv run python -m scripts.ingest

uv run python -m tests.test_core

uv run python -m evaluation.evaluate

uv run uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Alternative Quick Start with `pip`

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m scripts.ingest

python -m tests.test_core

python -m evaluation.evaluate

uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 19. Grounding and Abstention

The assistant is designed to answer only from the supplied Renata PLC leaflets.

When sufficient evidence is available:

```text
Retrieved Evidence
        |
        v
Generated Answer
        |
        v
Grounding Validation
        |
        v
Answer + Citations
```

When sufficient evidence is not available:

```text
Retrieved Evidence
        |
        v
Generated Answer
        |
        v
Grounding Validation
        |
        v
Unsupported
        |
        v
Abstain
```

This prevents the assistant from presenting unsupported information as if it came from the supplied leaflets.

---

# 20. Example Questions

The following questions can be used for manual validation:

```text
What is Rolac and what is it used for?
```

```text
What is the usual adult dose of Rolip?
```

```text
Use in pregnancy and breast-feeding
```

```text
What are the contraindications?
```

```text
What are the side effects?
```

```text
What information is available about this medicine?
```

The assistant should provide citation-supported answers when the information exists in the supplied documents and abstain when the documents do not provide sufficient evidence.

---

# 21. Troubleshooting

## `ModuleNotFoundError: No module named 'app'`

Make sure commands are executed from the project root.

For example:

```powershell
cd "D:\path\to\Renata-PLC-Leaflet-Assistant"
```

Then run:

```powershell
uv run python -m scripts.ingest
```

or:

```powershell
python -m scripts.ingest
```

---

## Groq API Error

Check that `.env` contains a valid API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Also verify that `GROQ_MODEL` contains a model available to your Groq account.

---

## Retrieval Results Are Incorrect

Rebuild the vector database:

```powershell
uv run python -m scripts.ingest
```

Then run:

```powershell
uv run python -m evaluation.evaluate
```

If retrieval is still incorrect, inspect the retrieval/ranking layer before changing the LLM prompt.

---

# 22. Summary

The Renata PLC Leaflet Assistant combines:

* **FastAPI** for the application interface
* **ChromaDB** for vector storage and retrieval
* **`sentence-transformers/all-MiniLM-L6-v2`** for embeddings
* **Groq + LangChain `ChatGroq`** for LLM generation
* Hybrid retrieval/ranking for improved evidence selection
* Citation validation
* Grounding validation
* Automated tests
* Automated evaluation

The complete application can be started with:

```powershell
uv sync
uv run python -m scripts.ingest
uv run python -m tests.test_core
uv run python -m evaluation.evaluate
uv run uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The result is a document-grounded assistant designed to answer questions from the supplied Renata PLC medicine leaflets while providing traceable supporting evidence.
