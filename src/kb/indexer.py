"""Bring the index up to date: re-embed only new or changed files, drop deleted ones."""

import time
from dataclasses import dataclass
from pathlib import Path

from . import keyword, store
from .chunker import chunk_file, with_header
from .discover import discover
from .embedder import embed
from .manifest import file_hash, file_key, load_manifest, save_manifest


@dataclass(frozen=True)
class IndexStats:
    files: int
    changed: int
    deleted: int
    new_chunks: int
    total_chunks: int
    seconds: float


def index_projects(root: Path, include: frozenset[str] | None) -> IndexStats:
    start = time.time()
    files = {file_key(f): f for f in discover(root, include)}

    old = load_manifest()
    new = {key: file_hash(f) for key, f in files.items()}
    deleted = [key for key in old if key not in new]
    changed = [key for key in new if old.get(key) != new[key]]

    # Remove old chunks first: an edited file may now have fewer chunks than before.
    # A file that was never indexed has nothing to remove.
    for key in deleted + [key for key in changed if key in old]:
        project, path = key.split("/", 1)
        store.delete_file(project, path)
        keyword.delete_file(project, path)

    chunks = [chunk for key in changed for chunk in chunk_file(files[key])]
    if chunks:
        store.add_chunks(chunks, embed([with_header(c) for c in chunks]))
        keyword.add_chunks(chunks)

    # Saved last: if the run crashes, the next run redoes the same work
    save_manifest(new)
    return IndexStats(
        files=len(new),
        changed=len(changed),
        deleted=len(deleted),
        new_chunks=len(chunks),
        total_chunks=store.count(),
        seconds=round(time.time() - start, 1),
    )
