"""Save chunks and their vectors in Chroma, and search them."""

from pathlib import Path

import chromadb

from .chunker import Chunk

DB_PATH = Path(__file__).resolve().parents[2] / "chroma_db"
BATCH_SIZE = 1000

_collection = chromadb.PersistentClient(path=str(DB_PATH)).get_or_create_collection(
    name="chunks",
    metadata={"hnsw:space": "cosine"},
)


def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.project}/{chunk.path}:{chunk.start_line}-{chunk.end_line}"


def add_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        _collection.upsert(
            ids=[chunk_id(c) for c in batch],
            embeddings=vectors[i:i + BATCH_SIZE],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "project": c.project,
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in batch
            ],
        )


def delete_file(project: str, path: str) -> None:
    _collection.delete(where={"$and": [{"project": project}, {"path": path}]})


def file_chunks(project: str, path: str) -> list[dict]:
    """Every chunk of one file, in line order."""
    result = _collection.get(where={"$and": [{"project": project}, {"path": path}]})
    chunks = [
        {"id": id_, "text": doc, **meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    return sorted(chunks, key=lambda chunk: chunk["start_line"])


def count() -> int:
    return _collection.count()


def search(query_vector: list[float], k: int = 5, project: str | None = None) -> list[dict]:
    """Closest chunks by meaning. Smaller distance is closer."""
    result = _collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"project": project} if project else None,
    )
    return [
        {"id": id_, "text": doc, "distance": dist, **meta}
        for id_, doc, dist, meta in zip(
            result["ids"][0],
            result["documents"][0],
            result["distances"][0],
            result["metadatas"][0],
        )
    ]
