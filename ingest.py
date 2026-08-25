"""Data ingestion pipeline (Week 3 of the plan).

Reads every .txt/.md/.pdf/.docx file in the documents folder, splits it
into chunks (markdown-heading sections first, then paragraph merging with
a small overlap between neighboring chunks), embeds each chunk with the
local embedding model, and stores chunk + vector in SQLite. Re-running
rebuilds the database.
"""

import re
import sys
from pathlib import Path

import docx
import pypdf

import config
import db
import llm

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def read_document(path: Path) -> str:
    """Extract plain text from a document, joining units with blank lines
    so chunk_text() can split on them."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = pypdf.PdfReader(path)
        pages = (page.extract_text() or "" for page in reader.pages)
        return "\n\n".join(p.strip() for p in pages if p.strip())
    if suffix == ".docx":
        document = docx.Document(path)
        return "\n\n".join(
            p.text.strip() for p in document.paragraphs if p.text.strip()
        )
    return path.read_text(encoding="utf-8")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text at markdown headings into (heading, body) pairs. Text
    without headings becomes a single section with an empty heading."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body_lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if body_lines:
                sections.append((heading, "\n".join(body_lines)))
                body_lines = []
            heading = line.lstrip("#").strip()
        else:
            body_lines.append(line)
    if body_lines:
        sections.append((heading, "\n".join(body_lines)))
    return sections


def _overlap_tail(text: str, max_chars: int) -> str:
    """Last full sentence(s) of `text`, at most max_chars long."""
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip()
        if len(candidate) > max_chars:
            break
        tail = candidate
    return tail


def chunk_text(
    text: str,
    max_chars: int = config.MAX_CHUNK_CHARS,
    overlap_chars: int = config.CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split into markdown-heading sections, then greedily merge paragraphs
    up to max_chars. Each chunk starts with its section heading (extra
    topical signal for the embedding), and neighboring chunks of the same
    section share an overlap so boundary information is not lost."""

    def finish(heading: str, chunk: str) -> str:
        return f"{heading}\n\n{chunk}" if heading else chunk

    chunks: list[str] = []
    for heading, body in split_sections(text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > max_chars:
                chunks.append(finish(heading, current))
                tail = _overlap_tail(current, overlap_chars)
                current = f"{tail}\n\n{para}" if tail else para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(finish(heading, current))
    return chunks


def ingest() -> None:
    doc_files = sorted(
        p for p in config.DOCUMENTS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_files:
        sys.exit(f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) "
                 f"found in {config.DOCUMENTS_DIR}")

    print(f"Loading embedding model ({config.EMBEDDING_MODEL_ALIAS})...")
    llm.init_embedding_model()

    conn = db.get_connection()
    db.clear_documents(conn)

    total = 0
    for path in doc_files:
        text = read_document(path)
        chunks = chunk_text(text)
        print(f"  {path.name}: {len(chunks)} chunks")
        embeddings = llm.embed_texts(chunks)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            db.insert_chunk(conn, path.name, idx, chunk, embedding)
        total += len(chunks)

    conn.commit()
    print(f"Done: {total} chunks stored in {config.DB_PATH.name} "
          f"({db.count_chunks(conn)} rows in DB).")
    conn.close()


if __name__ == "__main__":
    ingest()
