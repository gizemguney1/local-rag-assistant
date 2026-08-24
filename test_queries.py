"""Functional test script (Week 5 of the plan).

Runs a fixed set of questions through the assistant:
- answerable questions (the answer exists in the documents)
- an unanswerable question (the assistant should say it doesn't know)
- edge cases (empty / very general input)

Prints each answer plus timing so results can be recorded in the report.
"""

import sys
import time

import llm
from main import answer_query

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSWERABLE = [
    "What are the three steps of RAG?",
    "How do I install the Foundry Local Python SDK?",
    "What is cosine similarity used for?",
    "Why is SQLite a good choice for local storage?",
    "What temperature is recommended for factual Q&A?",
]

UNANSWERABLE = [
    "What is the capital of France?",
    "Who won the 2022 World Cup?",
]

EDGE_CASES = [
    "a",            # near-empty query
    "Tell me everything.",  # very general question
]


def run(label: str, questions: list[str]) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    for q in questions:
        start = time.perf_counter()
        answer = answer_query(q, verbose=True)
        elapsed = time.perf_counter() - start
        print(f"\nQ: {q}\nA: {answer}\n({elapsed:.1f}s)")


if __name__ == "__main__":
    print("Loading models...")
    llm.init_embedding_model()
    llm.init_chat_model()
    run("ANSWERABLE (should give grounded answers with sources)", ANSWERABLE)
    run("UNANSWERABLE (should say it doesn't know)", UNANSWERABLE)
    run("EDGE CASES", EDGE_CASES)
