"""The search strategies the agent can route a question to."""

import re

from src.kb import store
from src.kb.retrieval import search_chunks

STRATEGIES = {
    "hybrid search": "meaning and exact words together; a good default",
    "semantic search": "by meaning only; for how/why or conceptual questions",
    "keyword search": "exact words only; for names, tables, columns and config keys",
    "symbol lookup": "find where a function, class or file is defined; for identifiers like get_orders",
    "read whole file": "take the best-matching file and read all of it; for questions about one file",
}

MAX_FILE_CHUNKS = 12
# snake_case, camelCase, PascalCase and file names such as query.flux
IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[\w-]+\.\w+)\b"
)


def symbol_lookup(query: str, project: str | None, k: int) -> list[dict]:
    symbols = IDENTIFIER.findall(query) or query.split()
    hits = search_chunks(" ".join(symbols), k=k * 2, mode="keyword", project=project)

    def defines_symbol(hit: dict) -> bool:
        return any(
            re.search(rf"\b(def|class|function|const|let|var)\s+{re.escape(s)}\b", hit["text"])
            for s in symbols
        )

    return sorted(hits, key=lambda hit: not defines_symbol(hit))[:k]


def read_whole_file(query: str, project: str | None) -> list[dict]:
    best = search_chunks(query, k=1, project=project)
    if not best:
        return []
    return store.file_chunks(best[0]["project"], best[0]["path"])[:MAX_FILE_CHUNKS]


def run_strategy(name: str, query: str, project: str | None, k: int) -> list[dict]:
    if name == "semantic search":
        return search_chunks(query, k=k, mode="vector", project=project)
    if name == "keyword search":
        return search_chunks(query, k=k, mode="keyword", project=project)
    if name == "symbol lookup":
        return symbol_lookup(query, project, k)
    if name == "read whole file":
        return read_whole_file(query, project)
    return search_chunks(query, k=k, mode="hybrid", project=project)
