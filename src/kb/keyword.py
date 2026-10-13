"""Keyword (BM25) search over chunks, using SQLite's built-in FTS5."""

import re
import sqlite3

from .chunker import Chunk, with_header
from .store import DB_PATH, chunk_id

_db = sqlite3.connect(DB_PATH / "keyword.sqlite3")
_db.execute(
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        id UNINDEXED, project UNINDEXED, path UNINDEXED,
        start_line UNINDEXED, end_line UNINDEXED, text UNINDEXED,
        body,
        tokenize = 'porter unicode61'
    )
    """
)


def add_chunks(chunks: list[Chunk]) -> None:
    rows = [
        (chunk_id(c), c.project, c.path, c.start_line, c.end_line, c.text, with_header(c))
        for c in chunks
    ]
    with _db:
        _db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def delete_file(project: str, path: str) -> None:
    with _db:
        _db.execute("DELETE FROM chunks WHERE project = ? AND path = ?", (project, path))


def search(question: str, k: int = 5) -> list[dict]:
    words = re.findall(r"\w+", question)
    if not words:
        return []
    match = " OR ".join(f'"{word}"' for word in words)
    rows = _db.execute(
        """
        SELECT id, project, path, start_line, end_line, text
        FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?
        """,
        (match, k),
    ).fetchall()
    keys = ("id", "project", "path", "start_line", "end_line", "text")
    return [dict(zip(keys, row)) for row in rows]
