# Local RAG Assistant (Foundry Local)

A fully offline document Q&A assistant built with **Microsoft Foundry Local** and the
**Retrieval-Augmented Generation (RAG)** pattern, following the one-month summer school
plan in `Summer School Foundry Local Plan.docx`.

The assistant answers questions about a small local document collection by:
1. **Retrieve** — embedding the question and finding the most similar document chunks
   in a SQLite database (cosine similarity, brute-force scan).
2. **Augment** — pasting those chunks into the prompt as labeled context.
3. **Generate** — asking a local LLM (via Foundry Local) to answer *only* from that
   context, citing the source document, or saying it doesn't know.

Everything runs on-device: no cloud account, no internet needed at question time.

## Architecture

```
User (CLI) ──> main.py ──> retrieval.py ──> SQLite (rag.db: chunks + embeddings)
                  │                             ▲
                  │                             │ ingest.py (chunk + embed docs)
                  └──> llm.py ──> Foundry Local server (OpenAI-compatible API)
                                    ├─ qwen3-embedding-0.6b  (embeddings)
                                    └─ phi-3.5-mini          (chat answers)
```

| File | Role |
|---|---|
| `config.py` | Model aliases, retrieval/chunking settings, system prompt |
| `db.py` | SQLite data layer (`documents` table) |
| `llm.py` | Foundry Local model management, embeddings + chat calls |
| `ingest.py` | Ingestion pipeline: read docs → chunk → embed → store |
| `retrieval.py` | `get_top_chunks(query)` — cosine-similarity search |
| `main.py` | CLI interface and `answer_query()` orchestration |
| `app.py` | Streamlit web UI (chat + retrieved-context inspector) |
| `test_queries.py` | Functional test suite (answerable / unanswerable / edge cases) |
| `documents/` | The knowledge base (plain `.txt`/`.md` files) |

## Setup

1. Install [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/):
   `winget install Microsoft.FoundryLocal` (Windows) or `brew tap microsoft/foundrylocal && brew install foundrylocal` (macOS).
2. Install Python dependencies: `pip install -r requirements.txt`
3. Build the knowledge base (downloads the embedding model on first run):
   `python ingest.py`
4. Ask questions:
   - Interactive CLI: `python main.py`
   - One-shot: `python main.py "What is Foundry Local?"`
   - Show retrieved chunks: add `--verbose`
   - Web UI: `streamlit run app.py` (chat interface with a "Retrieved context"
     expander showing the chunks and similarity scores behind each answer)

To change the knowledge base, drop `.txt` or `.md` files into `documents/` and re-run
`python ingest.py`.

## Testing

`python test_queries.py` runs a fixed question set and prints answers with timings:
answerable questions should come back grounded with a source mention; out-of-scope
questions should get "I don't have that information in my documents."

## Design decisions & limitations

- **Brute-force vector search** over all rows is used instead of a vector index —
  simple and fast enough for a small knowledge base (hundreds of chunks).
- **Embeddings stored as JSON text** in SQLite for portability and easy debugging.
- A **similarity floor** (`MIN_SIMILARITY` in `config.py`) prevents irrelevant chunks
  from reaching the model, so off-topic questions fail fast without an LLM call.
- Small models are chosen for speed on laptop hardware; answers from `phi-3.5-mini`
  are decent but not comparable to large cloud models.
