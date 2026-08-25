# Test Report — Local RAG Assistant

**Date:** 2026-08-24 (initial round) · **updated 2026-08-25** after post-plan upgrades
**Phase:** Week 5 — System Testing & Evaluation (per *Summer School Foundry Local Plan.docx*), extended with retrieval evaluation of the upgraded system
**Result: 9 / 9 functional test cases passed · retrieval hit rate 15/15 (100%) · follow-up questions 3/3 with query rewriting.**

## 1. Test environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| Python | 3.13.2 |
| Foundry Local CLI | 0.10.3 · `foundry-local-sdk` 1.2.4 |
| Hardware | NVIDIA GeForce RTX 2050 (CUDA + WebGPU execution providers registered) |
| Chat model | `phi-3.5-mini` (temperature 0.2, max_tokens 600) |
| Embedding model | `qwen3-embedding-0.6b` |
| Knowledge base | initial round: 5 documents → 5 chunks · current: **11 documents (.txt + .pdf) → 16 chunks** |
| Retrieval | top-3 chunks by cosine similarity, minimum score 0.30 |
| Chunking | markdown-heading sections, paragraph merging ≤1200 chars, 120-char sentence overlap |

Test drivers: `python test_queries.py` (end-to-end answers through `answer_query()`,
the same code path as the CLI and the Streamlit UI) and `python evaluate.py`
(retrieval stage in isolation, plus rewrite on/off comparison).

## 2. Functional tests — answerable questions (initial round, 2026-08-24)

Expected: grounded answer with source citation.

| # | Question | Top retrieved doc (score) | Answer correct? | Source cited? | Time |
|---|---|---|---|---|---|
| 1 | What are the three steps of RAG? | 01_rag_basics.txt (0.782) | ✅ Retrieve / Augment / Generate | ✅ | 9.9 s |
| 2 | How do I install the Foundry Local Python SDK? | 02_foundry_local.txt (0.726) | ✅ `pip install foundry-local-sdk` | ✅ | 5.4 s |
| 3 | What is cosine similarity used for? | 03_embeddings.txt (0.593) | ✅ semantic comparison of embeddings | ✅ | 12.7 s |
| 4 | Why is SQLite a good choice for local storage? | 04_sqlite.txt (0.747) | ✅ serverless, single file, zero setup | ✅ | 11.5 s |
| 5 | What temperature is recommended for factual Q&A? | 05_prompt_engineering.txt (0.652) | ✅ "around 0.2" | ✅ | 4.3 s |

**Observation:** retrieval ranked the correct document first in **5 of 5** cases;
every answer named its source document.

## 3. Functional tests — unanswerable questions (initial round)

Expected: refuse, do not fabricate.

| # | Question | Behavior | Time |
|---|---|---|---|
| 6 | What is the capital of France? | One weakly-related chunk (0.348) passed the threshold, but the model correctly said the context does not contain the answer — it did **not** say "Paris" | 2.6 s |
| 7 | Who won the 2022 World Cup? | No chunk reached the 0.30 similarity floor → fixed fallback answer returned **without an LLM call** | 0.1 s |

**Observation:** the two safety layers both work — the similarity floor rejects
fully off-topic queries instantly, and the system prompt stops the model from
using outside knowledge when a marginal chunk slips through. Re-checked after
the knowledge base grew to 16 chunks: the World Cup question is still rejected
below the similarity floor.

## 4. Functional tests — edge cases (initial round)

| # | Input | Behavior | Time |
|---|---|---|---|
| 8 | `a` (near-empty query) | Explained no specific question was asked, invited the user to ask one; no crash, no fabrication | 20.3 s |
| 9 | "Tell me everything." | Declined to answer comprehensively; briefly listed the topics the documents cover, with sources | 9.5 s |

## 5. Retrieval evaluation (2026-08-25, `evaluate.py`, 16-chunk knowledge base)

The retrieval stage was scored in isolation against 15 labeled questions
(each tagged with the document that holds its answer, 1–2 questions per
document, including the PDF):

| Metric | Result |
|---|---|
| Hit rate (correct doc in top-3) | **15/15 = 100%** |
| Top-1 rate (correct doc ranked first) | 14/15 = 93% |
| Mean reciprocal rank (MRR) | 0.967 |

The single non-top-1 case ("Why does the assistant sometimes refuse to
answer?", expected `11_project_faq.pdf`, found at rank 2) loses narrowly to
the prompt-engineering course note, which legitimately covers the same topic.

**Chunk-overlap tuning caught by the metric:** the first overlap setting
(200 chars ≈ 15%) silently degraded ranking — top-1 dropped 93% → 87% and MRR
0.967 → 0.922, because on short documents a large repeated tail dilutes the
chunk embedding. Reducing the overlap to 120 chars (≈10%) restored baseline
ranking exactly while keeping the boundary-information benefit. This is the
measure → change → re-measure loop the eval script exists for.

## 6. Conversational follow-up questions (2026-08-25)

Follow-up questions are rewritten into standalone questions from the recent
conversation before retrieval (`rewrite_query()`, one extra LLM call capped at
80 tokens). Measured effect on 3 follow-up cases:

| Follow-up question (with prior turn) | Raw query | Rewritten query |
|---|---|---|
| "Why is it useful?" (after chunk overlap) | **MISS** | rank 1 |
| "How do I switch to a smaller one?" (after model names) | MISS | rank 2 |
| "How do I install its Python SDK?" (after Foundry Local) | rank 2 | rank 1 |
| **Hit rate** | **1/3** | **3/3** |

Standalone questions pass through unchanged (verified), so `test_queries.py`
results and single-question latency are unaffected. The Streamlit UI shows the
rewritten question as *Interpreted as: "…"* for transparency.

## 7. Performance

- Initial round: full answers in **0.1–20.3 s** (typical 4–13 s).
- **Streaming** (added 2026-08-25): the first words of an answer now appear
  after **~3 s**; a full answer completes in ~6–13 s while text is visibly
  flowing. Follow-up questions add the rewrite call: first words ~3–6 s,
  ~8 s total in clean measurements.
- Off-topic questions still cost almost nothing (0.1 s) thanks to the
  similarity floor.
- The plan's aspirational target was ~1–3 s/question; with streaming, the
  *perceived* wait now meets it even though full generation takes longer.
- **Measurement caveat:** running the CLI and the Streamlit server at the same
  time loads every model twice and exceeds the 4 GB GPU, slowing answers 3–4×.
  All timings above were taken with a single process using the models.

## 8. Issues found & fixes

| Issue | Status |
|---|---|
| Em-dashes in model output rendered as `â€”` when stdout is redirected (Windows ANSI codepage) | **Fixed** — CLI and test scripts force UTF-8 stdout |
| Response time above the plan's 1–3 s target | **Mitigated** — streaming makes first words appear in ~3 s; optimizations documented in the FAQ |
| Single-page multi-topic FAQ PDF was never retrieved (all five topics in one chunk blurred its embedding) | **Fixed** — one FAQ entry per page → one chunk per topic; question then retrieved at rank 1 |
| 200-char chunk overlap degraded top-1 rate 93% → 87% | **Fixed** — overlap tuned to 120 chars via `evaluate.py`; baseline ranking restored |
| Follow-up questions ("Why is it useful?") retrieved nothing | **Fixed** — conversational query rewriting; follow-up hit rate 1/3 → 3/3 |

## 9. Conclusion

The assistant meets all functional goals of the plan and now exceeds them:
it answers in-scope questions correctly with source citations (100% retrieval
hit rate on the labeled set), refuses out-of-scope questions without
hallucinating, handles malformed input gracefully, ingests real-world PDF and
DOCX documents, streams answers so the first words appear in ~3 s, and
understands follow-up questions in conversation — fully offline. Suggested
demo script: one grounded answer, one follow-up question showing the
*Interpreted as* rewrite, one refusal, and the Streamlit "Retrieved context"
panel to show retrieval in action.
