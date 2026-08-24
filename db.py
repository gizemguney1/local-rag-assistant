"""SQLite data layer: stores document chunks and their embedding vectors.

The embedding is stored as a JSON-serialized list of floats (simple and
portable; fine for a small local knowledge base).
"""

import json
import sqlite3

import config


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    return conn


def clear_documents(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM documents")
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    source: str,
    chunk_idx: int,
    content: str,
    embedding: list[float],
) -> None:
    conn.execute(
        "INSERT INTO documents (source, chunk_idx, content, embedding) VALUES (?, ?, ?, ?)",
        (source, chunk_idx, content, json.dumps(embedding)),
    )


def fetch_all_chunks(conn: sqlite3.Connection) -> list[dict]:
    """Return every chunk with its embedding deserialized back to floats."""
    rows = conn.execute(
        "SELECT id, source, chunk_idx, content, embedding FROM documents"
    ).fetchall()
    return [
        {
            "id": row[0],
            "source": row[1],
            "chunk_idx": row[2],
            "content": row[3],
            "embedding": json.loads(row[4]),
        }
        for row in rows
    ]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
