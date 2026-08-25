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


def rewrite_query(history: list[tuple[str, str]], question: str) -> str:
    """Rewrite a follow-up question into a standalone one so retrieval (and
    the answer prompt, which also never sees the conversation) can understand
    it. Returns the question unchanged when there is no history yet."""
    if not history:
        return question

    recent = history[-config.REWRITE_HISTORY_TURNS:]
    convo = "\n".join(
        f"User: {q}\nAssistant: {a[:300]}" for q, a in recent
    )
    rewritten = llm.chat(
        config.REWRITE_PROMPT,
        f"Conversation:\n{convo}\n\nNew question: {question}\n\n"
        "Standalone question:",
        max_tokens=80,
    )
    first_line = next(
        (line.strip().strip('"') for line in rewritten.splitlines()
         if line.strip()),
        "",
    )
    # A tiny model can misbehave; fall back to the original question rather
    # than search for something malformed.
    if not first_line or len(first_line) > 300:
        return question
    return first_line


def build_user_prompt(user_question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Answer the question using only the context above."
    )


def answer_query_stream(
    user_question: str,
    verbose: bool = False,
    history: list[tuple[str, str]] | None = None,
):
    """Return an iterator over the answer text: rewriting (if there is
    history) and retrieval run immediately, then the LLM's text arrives
    piece by piece."""
    search_query = rewrite_query(history or [], user_question)
    if verbose and search_query != user_question:
        print(f"  [rewritten] {search_query}")

    chunks = retrieval.get_top_chunks(search_query)

    if verbose:
        for c in chunks:
            print(f"  [retrieved] {c['source']}#{c['chunk_idx']} "
                  f"(score {c['score']:.3f})")

    if not chunks:
        return iter([FALLBACK_ANSWER])

    return llm.chat_stream(
        config.SYSTEM_PROMPT, build_user_prompt(search_query, chunks)
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
          "Type a question, or 'quit' to exit. Follow-up questions are "
          "understood in context.\n")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        stream = answer_query_stream(question, verbose=verbose, history=history)
        print("\nAssistant: ", end="", flush=True)
        pieces = []
        for piece in stream:
            print(piece, end="", flush=True)
            pieces.append(piece)
        print("\n")
        history.append((question, "".join(pieces).strip()))

    print("Goodbye!")


if __name__ == "__main__":
    main()
