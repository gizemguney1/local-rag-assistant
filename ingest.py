"""Data ingestion pipeline (Week 3 of the plan).

Reads every .txt/.md/.pdf/.docx file in the documents folder, splits it
into paragraph-based chunks, embeds each chunk with the local embedding
model, and stores chunk + vector in SQLite. Re-running rebuilds the
database.
"""

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


def chunk_text(text: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[str]:
    """Split text on blank lines, then greedily merge paragraphs up to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
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
