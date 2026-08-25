"""Retrieval evaluation: measures whether the correct source document is
retrieved for a set of labeled questions.

Unlike test_queries.py (which inspects end-to-end answers), this script
scores the retrieval stage in isolation: for each question we know which
document holds the answer, so we can compute hit rate (correct document
in the top-k results), top-1 rate, and mean reciprocal rank (MRR).

A second section evaluates follow-up questions with and without
conversational query rewriting, quantifying what rewrite_query() adds.

Usage: python evaluate.py [--skip-followups]
"""

import sys

import config
import llm
import retrieval
from main import rewrite_query

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (question, source document that contains the answer)
LABELED_QUESTIONS = [
    ("What are the three steps of RAG?", "01_rag_basics.txt"),
    ("How does RAG compare with fine-tuning?", "01_rag_basics.txt"),
    ("How do I install the Foundry Local Python SDK?", "02_foundry_local.txt"),
    ("Which command-line tool does Foundry Local provide?", "02_foundry_local.txt"),
    ("What is cosine similarity used for?", "03_embeddings.txt"),
    ("Why is SQLite a good choice for local storage?", "04_sqlite.txt"),
    ("What temperature is recommended for factual Q&A?", "05_prompt_engineering.txt"),
    ("When should I switch from brute-force search to an ANN index?", "06_vector_databases.txt"),
    ("What is an HNSW graph used for?", "06_vector_databases.txt"),
    ("How much chunk overlap is recommended?", "07_chunking_strategies.txt"),
    ("What are the advantages of running an LLM locally instead of in the cloud?", "08_local_vs_cloud_llms.txt"),
    ("What is hit rate in RAG evaluation?", "09_rag_evaluation.txt"),
    ("What does st.cache_resource do in Streamlit?", "10_streamlit_basics.txt"),
    ("Why does the assistant sometimes refuse to answer?", "11_project_faq.pdf"),
    ("How can I make the assistant answer faster?", "11_project_faq.pdf"),
]

# (conversation history, follow-up question, expected source document)
FOLLOW_UPS = [
    (
        [("What is chunk overlap?",
          "Chunk overlap means neighboring chunks share the last one or two "
          "sentences so boundary information is not lost. Typical overlap is "
          "10 to 20 percent of the chunk size.")],
        "Why is it useful?",
        "07_chunking_strategies.txt",
    ),
    (
        [("Which models does the assistant use?",
          "Chat answers come from phi-3.5-mini and embeddings from "
          "qwen3-embedding-0.6b, both running offline through Foundry Local.")],
        "How do I switch to a smaller one?",
        "11_project_faq.pdf",
    ),
    (
        [("What is Foundry Local?",
          "Foundry Local is a local AI runtime and SDK from Microsoft for "
          "running language models fully on-device with a curated catalog "
          "of optimized models.")],
        "How do I install its Python SDK?",
        "02_foundry_local.txt",
    ),
]


def rank_of(source: str, chunks: list[dict]) -> int | None:
    """1-based rank of the first chunk from `source`, or None if absent."""
    for i, c in enumerate(chunks, start=1):
        if c["source"] == source:
            return i
    return None


def evaluate_retrieval() -> None:
    print("=" * 72)
    print("RETRIEVAL EVALUATION (labeled questions)")
    print("=" * 72)
    hits = top1 = 0
    reciprocal_ranks = []
    for question, expected in LABELED_QUESTIONS:
        chunks = retrieval.get_top_chunks(question)
        r = rank_of(expected, chunks)
        got = f"rank {r}" if r else "MISS"
        top = f"{chunks[0]['source']} ({chunks[0]['score']:.3f})" if chunks else "-"
        print(f"  [{got:6}] {question}\n           expected {expected}, top hit {top}")
        if r:
            hits += 1
            reciprocal_ranks.append(1 / r)
            if r == 1:
                top1 += 1
        else:
            reciprocal_ranks.append(0.0)
    n = len(LABELED_QUESTIONS)
    print(f"\n  Hit rate (top-{config.TOP_K}): {hits}/{n} = {hits / n:.0%}")
    print(f"  Top-1 rate:            {top1}/{n} = {top1 / n:.0%}")
    print(f"  Mean reciprocal rank:  {sum(reciprocal_ranks) / n:.3f}")


def evaluate_followups() -> None:
    print("\n" + "=" * 72)
    print("FOLLOW-UP QUESTIONS (raw vs. rewritten query)")
    print("=" * 72)
    raw_hits = rewritten_hits = 0
    for history, question, expected in FOLLOW_UPS:
        raw_rank = rank_of(expected, retrieval.get_top_chunks(question))
        rewritten = rewrite_query(history, question)
        new_rank = rank_of(expected, retrieval.get_top_chunks(rewritten))
        raw_hits += raw_rank is not None
        rewritten_hits += new_rank is not None
        print(f"  Q: {question}")
        print(f"     raw:       {'rank ' + str(raw_rank) if raw_rank else 'MISS'}")
        print(f"     rewritten: {'rank ' + str(new_rank) if new_rank else 'MISS'}"
              f"  ({rewritten})")
    n = len(FOLLOW_UPS)
    print(f"\n  Hit rate without rewriting: {raw_hits}/{n}")
    print(f"  Hit rate with rewriting:    {rewritten_hits}/{n}")


if __name__ == "__main__":
    print("Loading embedding model...")
    llm.init_embedding_model()
    evaluate_retrieval()
    if "--skip-followups" not in sys.argv:
        print("\nLoading chat model (for query rewriting)...")
        llm.init_chat_model()
        evaluate_followups()
