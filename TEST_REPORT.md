# Test Report — Local RAG Assistant

**Date:** 2026-08-24
**Phase:** Week 5 — System Testing & Evaluation (per *Summer School Foundry Local Plan.docx*)
**Result: 9 / 9 test cases passed.**

## 1. Test environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| Python | 3.13.2 |
| Foundry Local CLI | 0.10.3 · `foundry-local-sdk` 1.2.4 |
| Hardware | NVIDIA GeForce RTX 2050 (CUDA + WebGPU execution providers registered) |
| Chat model | `phi-3.5-mini` (temperature 0.2, max_tokens 600) |
| Embedding model | `qwen3-embedding-0.6b` |
| Knowledge base | 5 course-note documents → 5 chunks in `rag.db` (SQLite) |
| Retrieval | top-3 chunks by cosine similarity, minimum score 0.30 |

Test driver: `python test_queries.py` (all questions run end-to-end through
`answer_query()`, the same code path as the CLI and the Streamlit UI).

## 2. Answerable questions — expected: grounded answer with source citation

| # | Question | Top retrieved doc (score) | Answer correct? | Source cited? | Time |
|---|---|---|---|---|---|
| 1 | What are the three steps of RAG? | 01_rag_basics.txt (0.782) | ✅ Retrieve / Augment / Generate | ✅ | 9.9 s |
| 2 | How do I install the Foundry Local Python SDK? | 02_foundry_local.txt (0.726) | ✅ `pip install foundry-local-sdk` | ✅ | 5.4 s |
| 3 | What is cosine similarity used for? | 03_embeddings.txt (0.593) | ✅ semantic comparison of embeddings | ✅ | 12.7 s |
| 4 | Why is SQLite a good choice for local storage? | 04_sqlite.txt (0.747) | ✅ serverless, single file, zero setup | ✅ | 11.5 s |
| 5 | What temperature is recommended for factual Q&A? | 05_prompt_engineering.txt (0.652) | ✅ "around 0.2" | ✅ | 4.3 s |

**Observation:** retrieval ranked the correct document first in **5 of 5** cases;
every answer named its source document.

## 3. Unanswerable questions — expected: refuse, do not fabricate

| # | Question | Behavior | Time |
|---|---|---|---|
| 6 | What is the capital of France? | One weakly-related chunk (0.348) passed the threshold, but the model correctly said the context does not contain the answer — it did **not** say "Paris" | 2.6 s |
| 7 | Who won the 2022 World Cup? | No chunk reached the 0.30 similarity floor → fixed fallback answer returned **without an LLM call** | 0.1 s |

**Observation:** the two safety layers both work — the similarity floor rejects
fully off-topic queries instantly, and the system prompt stops the model from
using outside knowledge when a marginal chunk slips through.

## 4. Edge cases — expected: graceful handling

| # | Input | Behavior | Time |
|---|---|---|---|
| 8 | `a` (near-empty query) | Explained no specific question was asked, invited the user to ask one; no crash, no fabrication | 20.3 s |
| 9 | "Tell me everything." | Declined to answer comprehensively; briefly listed the topics the documents cover, with sources | 9.5 s |

## 5. Performance

- Response times ranged **0.1–20.3 s** (typical 4–13 s).
- The plan's aspirational target was ~1–3 s/question; we exceed it because
  `phi-3.5-mini` generates fairly long, well-formed answers. Acceptable for a
  course project. Known optimizations if needed: retrieve 2 chunks instead of 3,
  lower `max_tokens`, or switch to a smaller chat model (e.g. `qwen3-1.7b`).
- Off-topic questions cost almost nothing (0.1 s) thanks to the similarity floor.

## 6. Issues found & fixes

| Issue | Status |
|---|---|
| Em-dashes in model output rendered as `â€”` when stdout is redirected (Windows ANSI codepage) | **Fixed** — CLI and test scripts now force UTF-8 stdout |
| Response time above the plan's 1–3 s target | Accepted (see §5); optimizations documented |

## 7. Conclusion

The assistant meets all functional goals of the plan: it answers in-scope
questions correctly with source citations, refuses out-of-scope questions
without hallucinating, and handles malformed input gracefully — fully offline.
The project is ready for the final demo (suggested demo script: one grounded
answer, one refusal, and the Streamlit "Retrieved context" panel to show
retrieval in action).
