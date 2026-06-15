# Vectorless RAG Chatbot

A retrieval-augmented chatbot that answers questions over your documents
**without embeddings and without a vector database** — powered entirely by a
local, open-source LLM (via [Ollama](https://ollama.com)).

---

## What is "vectorless" RAG?

**Classic RAG** works like this:

```
docs → chunk → embedding model → vector DB → cosine-similarity search → top-k chunks → LLM answer
```

It needs an embedding model, a vector store (FAISS, Chroma, Pinacone…), and it
retrieves by *numeric similarity*, which is opaque and struggles with
relational or multi-hop questions ("which policy applies to remote staff who
also handle confidential data?").

**Vectorless RAG** removes the embedding + vector-DB layer entirely and uses the
**LLM itself as the retriever**:

```
docs → chunk → LLM writes a 1-line summary per chunk (a "table of contents")
question → LLM reads the summaries → reasons about which chunks are relevant → picks ids → LLM answers from those chunks
```

| | Classic (vector) RAG | Vectorless RAG (this project) |
|---|---|---|
| Embedding model | required | **none** |
| Vector database | required | **none** (just `index.json`) |
| Retrieval method | cosine similarity | **LLM reasoning over summaries** |
| Explainability | low (distance scores) | **high (model states its reason)** |
| Multi-hop / relational queries | weak | **strong** |
| Cost per query | cheap lookup | one extra LLM call |
| Scales to millions of chunks | yes | best for small/medium corpora or hierarchical trees |

The trade-off: retrieval costs an extra LLM call per question, so it shines on
small-to-medium knowledge bases (handbooks, product docs, policies, a codebase
folder) rather than web-scale corpora.

---

## Architecture

```
documents/*.md ──► ingest.py ──► index.json        (chunks + LLM summaries; NO vectors)
                                     │
question ─────────► retriever.py ────┤  LLM reads summaries, returns relevant chunk ids (JSON)
                                     │
                    rag.py ──────────┘  feeds full text of chosen chunks to the LLM → grounded, cited answer
                                     │
                    chatbot.py (CLI)  /  app.py (Streamlit UI)
```

| File | Responsibility |
|------|----------------|
| `config.py`    | Model name, Ollama host, chunk sizes |
| `llm.py`       | Minimal Ollama REST client (text + JSON modes) |
| `ingest.py`    | Split docs into chunks, generate per-chunk summaries, write `index.json` |
| `retriever.py` | **The vectorless core** — LLM selects relevant chunks by reasoning |
| `rag.py`       | Retrieve → build context → generate cited answer |
| `chatbot.py`   | Terminal chat loop |
| `app.py`       | Streamlit web chat UI with a retrieval trace |
| `documents/`   | Sample knowledge base (Acme Robotics handbook, FAQ, security policy) |

---

## Setup

### 1. Install Ollama and pull an open-source model

Download Ollama from <https://ollama.com>, then:

```powershell
ollama pull llama3.2        # ~2 GB, small & fast. Or: qwen2.5, mistral, phi3
ollama serve                # usually already running after install
```

Pick any model you like and point the project at it:

```powershell
$env:VRAG_MODEL = "qwen2.5"   # optional; defaults to llama3.2
```

### 2. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Build the index (one time)

```powershell
python ingest.py
```

This reads `documents/`, has the model summarize each chunk, and writes
`index.json`. Re-run whenever the documents change.

---

## Run the chatbot

**Terminal:**

```powershell
python chatbot.py
```

**Web UI:**

```powershell
streamlit run app.py
```

### Try these questions
- *"How many days of vacation do full-time employees get?"*
- *"How long does the warehouse robot run on a charge?"*
- *"What must I do if my laptop is stolen?"*
- *"A remote employee handles customer records — what rules apply to them?"*  ← multi-document, where vectorless reasoning helps.

After any terminal answer, type `sources` to see which chunks the model chose
and **why**. In the Streamlit UI, expand **Retrieval trace**.

---

## Use your own documents

Drop any `.txt` or `.md` files into `documents/`, delete `index.json`, and run
`python ingest.py` again. That's it.

---

## How retrieval actually works (the interesting part)

`retriever.py` builds a catalogue like:

```
# Document: employee_handbook.md
  - c0 [Working Hours]: Standard hours are 9–5 with flexible start times...
  - c2 [Paid Time Off]: Full-time staff accrue 20 vacation days plus 10 holidays...
# Document: security_policy.md
  - c8 [Data Classification]: Confidential data must be encrypted and on managed devices...
```

It sends this plus the question to the model in **JSON mode** and gets back:

```json
{ "chunk_ids": ["c2"], "reason": "c2 covers vacation accrual for full-time staff." }
```

Only the full text of the selected chunks is then passed to the answer prompt.
No vectors are ever computed — the "search" is the model reading summaries and
reasoning, exactly as a person would skim a table of contents.

---

## Limitations & next steps
- **Latency:** one extra LLM call per query. Fine locally; batch or cache for scale.
- **Large corpora:** flatten-and-list works up to a few hundred chunks. Beyond
  that, make the index *hierarchical* (document summaries → section summaries →
  chunks) and let the model descend the tree — true tree-search retrieval.
- **Robustness:** JSON mode keeps selection parseable; you can add a fallback
  that returns all chunks from a document if the model picks none.
