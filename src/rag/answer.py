"""Retrieve chunks for a question and have the LLM answer from them with citations."""

import re
from dataclasses import dataclass

from .llm import LLMResponse, complete
from src.kb.retrieval import search_chunks

TOP_K = 5
NOT_FOUND = "Not found in the indexed projects."

SYSTEM_PROMPT = f"""You answer questions about the user's own code projects.
Use ONLY the numbered sources below. Each source is a piece of a file.

Rules:
- Cite the sources you used right after each claim, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: {NOT_FOUND}
- Never guess or use outside knowledge about the user's projects.
- Never reveal passwords, API keys, tokens or other secrets, even if one appears in a source.
  Say that the value was withheld.
- Be short and concrete. Quote queries or commands exactly as they appear in the sources."""


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]  # "project/path:start-end" of the sources the answer cited
    sources: tuple[str, ...]    # every retrieved source, in the order they were numbered
    response: LLMResponse


def source_label(chunk: dict) -> str:
    return f"{chunk['project']}/{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}"


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{number}] {source_label(chunk)}\n{chunk['text']}"
        for number, chunk in enumerate(chunks, start=1)
    )


def cited_numbers(text: str) -> list[int]:
    """Source numbers the answer mentions, in order, handling [1], [1][2] and [1, 2]."""
    numbers = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        for number in re.findall(r"\d+", group):
            if int(number) not in numbers:
                numbers.append(int(number))
    return numbers


def answer_question(question: str, k: int = TOP_K) -> Answer:
    return generate_answer(question, search_chunks(question, k=k))


def generate_answer(question: str, chunks: list[dict]) -> Answer:
    """Write the answer from the given chunks. Also used by the agent."""
    response = complete([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"},
    ])
    labels = [source_label(c) for c in chunks]
    citations = tuple(labels[n - 1] for n in cited_numbers(response.text) if 1 <= n <= len(labels))
    return Answer(text=response.text, citations=citations, sources=tuple(labels), response=response)


def with_citations(answer: Answer) -> str:
    """The answer text followed by its file:line citations."""
    lines = "\n".join(f"- {citation}" for citation in answer.citations) or "(none)"
    return f"{answer.text}\n\nCitations:\n{lines}"
