"""Split a source file into fixed-size line windows."""

from dataclasses import dataclass

from .discover import SourceFile
from .safety import contains_secret

MAX_FILE_BYTES = 1_000_000
CHUNK_LINES = 60
OVERLAP_LINES = 15
MAX_CHUNK_CHARS = 2000  # the embedding model reads about 2000 chars, the rest is ignored


@dataclass(frozen=True)
class Chunk:
    project: str
    path: str           # rel_path
    start_line: int     # 1-based, inclusive
    end_line: int
    text: str


def chunk_file(file: SourceFile) -> list[Chunk]:
    if file.size > MAX_FILE_BYTES:
        return []
    try:
        text = file.path.read_text(encoding="utf-8")
    except UnicodeDecodeError:  # binary file
        return []
    if contains_secret(text):
        return []

    lines = text.splitlines()
    step = CHUNK_LINES - OVERLAP_LINES
    chunks = []
    for start in range(0, len(lines), step):
        window = lines[start:start + CHUNK_LINES]
        chunks.append(Chunk(
            project=file.project,
            path=file.rel_path,
            start_line=start + 1,
            end_line=start + len(window),
            text="\n".join(window)[:MAX_CHUNK_CHARS],
        ))
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks
