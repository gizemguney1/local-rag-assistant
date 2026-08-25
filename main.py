"""Local RAG Q&A assistant — CLI entry point (Week 4, Option A).

Usage:
    python ingest.py     # build/rebuild the knowledge base first
    python main.py       # then ask questions interactively
    python main.py "What is Foundry Local?"   # or one-shot
"""

import sys

import config
import db
import llm
import retrieval

# Model output may contain non-ASCII characters (em-dashes, quotes); Windows
# redirects stdout with the ANSI codepage by default, which mangles them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


FALLBACK_ANSWER = "I don't have that information in my documents."


def build_user_prompt(user_question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Answer the question using only the context above."
    )


def answer_query_stream(user_question: str, verbose: bool = False):
    """Return an iterator over the answer text: retrieval runs immediately,
    then the LLM's text arrives piece by piece."""
    chunks = retrieval.get_top_chunks(user_question)

    if verbose:
        for c in chunks:
            print(f"  [retrieved] {c['source']}#{c['chunk_idx']} "
                  f"(score {c['score']:.3f})")

    if not chunks:
        return iter([FALLBACK_ANSWER])

    return llm.chat_stream(
        config.SYSTEM_PROMPT, build_user_prompt(user_question, chunks)
    )


def answer_query(user_question: str, verbose: bool = False) -> str:
    return "".join(answer_query_stream(user_question, verbose)).strip()


def main() -> None:
    conn = db.get_connection()
    n_chunks = db.count_chunks(conn)
    conn.close()
    if n_chunks == 0:
        sys.exit("Knowledge base is empty. Run `python ingest.py` first.")

    verbose = "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    print("Loading local models (first run may download them)...")
    llm.init_embedding_model()
    llm.init_chat_model()

    if args:  # one-shot mode: question passed on the command line
        for piece in answer_query_stream(" ".join(args), verbose=verbose):
            print(piece, end="", flush=True)
        print()
        return

    print(f"\nLocal RAG assistant ready ({n_chunks} chunks indexed). "
          "Type a question, or 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        stream = answer_query_stream(question, verbose=verbose)
        print("\nAssistant: ", end="", flush=True)
        for piece in stream:
            print(piece, end="", flush=True)
        print("\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
