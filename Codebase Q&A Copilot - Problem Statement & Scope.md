# Codebase Q&A Copilot — Problem Statement & Scope

Oct 11, 2026 · @Satvik Singhal

## Problem

Knowledge I already wrote is hard to find again. My work spans many projects: Airflow DAGs, CPO work, AHU work, DeJoule work, Neptune queries and configs.

When I need to know how I did something before, such as "what query did I use for telemetry data in project X", I have to remember which repo it was in, grep for it and open files by hand. This gets slower as the number of projects grows.

## Goal and users

Build an agentic RAG copilot that indexes my projects directory and answers natural-language questions about it. Every answer cites the exact file and line it came from.

- **User:** me, on one machine. No multi-user support.
- **Purpose:** daily use, and a resume project that shows agentic RAG, tool use and evaluation.
- **Corpus:** everything in the projects directory: code, notebooks, SQL and other queries, configs and docs.
- **Models:** cloud LLM through OpenRouter for answers; Jev (TypeSafe AI) as the decision layer inside the agent.

## Ingestion and indexing (open)

Not decided yet. I will experiment on my own projects and pick the approach that retrieves best, measured with the evaluation set below.

Options to try:

- **Chunking:** code-aware (by function, class, notebook cell, SQL statement, config block, doc heading) vs. fixed-size windows with overlap.
- **Retrieval:** vector only vs. hybrid vector + keyword (BM25). Keyword search may matter for exact table and column names.
- **Chunk metadata:** project, path, language, symbol, line range, last modified.
- **Embeddings:** API model vs. local model.
- **Re-indexing:** full rebuild vs. incremental by file hash.

Fixed regardless of the choice: `.gitignore` is respected, and `.env` files and secrets are never indexed.

## Agentic retrieval loop

A LangGraph agent searches, judges what it found and loops until it has enough evidence. Jev makes the small decisions; the OpenRouter LLM only rewrites the query and writes the final answer.

&#91;embedded content: agent loop · 6 steps, 2 checks\]

If Jev judges the evidence thin, the agent re-routes and searches again; an answer reaches the user only after Jev confirms its citations support it.

Jev is a decision model, not a text generator: its primitives are binary classification, choice ranking and score ladders ([MindStudio](https://www.mindstudio.ai/blog/jev-reranker-rag/)).

| Step | Decided by | Jev primitive |
| --- | --- | --- |
| Resolve which project(s) the question is about | Jev | Choice ranking |
| Route to a strategy: semantic search, keyword search, symbol lookup, read a whole file | Jev | Choice ranking |
| Rerank retrieved chunks against the question | Jev | Score ladder |
| Decide whether there is enough evidence or search again (with a loop cap) | Jev | Binary |
| Write the answer with file:line citations | LLM (OpenRouter) | None (generation) |
| Check the answer is supported by its cited chunks; if not, retry or say "not found" | Jev | Binary |

Jev sits behind a `DecisionProvider` interface with an LLM fallback, so either can run each step.

## Interfaces

Two ways in: a CLI for the terminal and an MCP server so the copilot works inside Claude Code, Cursor and VS Code.

- **CLI:** `index <projects dir>`, `ask "<question>"`.
- **MCP tools:** `ask_codebase`, `search_code`, `list_projects`, `reindex`.

## Evaluation

A small test set over my own projects decides the ingestion choices and proves the agent works.

- **Test set:** about 50 hand-written questions, each with the file (and lines) that holds the answer.
- **Metrics:** retrieval recall@5, citation accuracy, answer correctness, p50 latency, cost per query.
- **Ingestion experiments:** run every chunking and retrieval option against the same test set.
- **Ablation:** Jev decision layer vs. LLM-as-judge at the same steps, compared on cost, latency and accuracy. This is the headline result for the resume.

## Out of scope for v1

- Editing or generating code: the copilot only reads.
- Git history questions ("what did this query look like last month"): planned for v2.
- Multiple users, cloud hosting, a web UI.
- Running code or connecting to live databases.

## Constraints and risks

- **Code leaves the machine.** Retrieved chunks are sent to OpenRouter. Work repos may fall under employer policy, so a per-project exclude list ships from day one.
- **Jev is new.** TypeSafe AI released it in September 2026, so its SDK and pricing may change. The `DecisionProvider` interface with an LLM fallback limits the risk.
- **Secrets.** `.env` files and anything matching a secret pattern are skipped before embedding or sending.

## Success criteria

- The answer cites the correct file for at least 80% of test questions.
- p50 response time under about 10 seconds.
- Re-indexing after a day's work takes under a minute.
- A written Jev vs. LLM-judge comparison with measured numbers.

## Open questions

- [ ] Which ingestion and indexing approach works best on my data?
- [ ] API or local embedding model?
- [ ] How large is the projects directory (repos, files)? This decides the vector store.
- [ ] Project name.
