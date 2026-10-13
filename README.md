# Codebase Q&A Copilot

Ask questions about your own projects in plain English and get short answers that cite the exact `file:line` range they came from. Typical questions: "What Flux query do I use to pull telemetry for one component?" or "Which SPARQL query lists all chillers?"

Everything is indexed locally (embeddings, vector store and keyword index). Only the retrieved snippets and your question are sent to an LLM through [OpenRouter](https://openrouter.ai).

## How it works

```
 projects ──► discover ──► chunk ──► embed (local) ──► Chroma (vectors)
                 │                       └──────────► SQLite FTS5 (BM25)
                 └─ skips secrets, binaries, .git, node_modules, ...

 question ──► agent (LangGraph) ──► hybrid retrieval ──► LLM answer + citations
```

**Indexing** ([src/kb/](src/kb))
- Finds files in each project, using `git ls-files` where a `.git` exists so ignored files stay out ([discover.py](src/kb/discover.py)).
- Splits files into 60-line windows with 15 lines of overlap ([chunker.py](src/kb/chunker.py)).
- Embeds each chunk, prefixed with its project and path, using `BAAI/bge-small-en-v1.5` on your machine ([embedder.py](src/kb/embedder.py)).
- Stores vectors in Chroma and the text in a SQLite FTS5 table for BM25 keyword search.
- Keeps a hash manifest, so re-indexing only touches new, changed or deleted files ([indexer.py](src/kb/indexer.py)).

**Retrieval** ([retrieval.py](src/kb/retrieval.py)): vector search and BM25 keyword search, merged with reciprocal rank fusion.

**Answering** ([src/rag/](src/rag)): the LLM may only use the numbered sources it is given, must cite them as `[1]`, `[2]`, and must reply "Not found in the indexed projects." when the sources don't contain the answer.

**Agent** ([src/agent/](src/agent)): by default a question goes through a LangGraph loop of up to 3 searches:

```
resolve_project → rewrite → route → retrieve → rerank → check_evidence
    → (thin evidence: back to rewrite) → answer → verify → (unsupported: back to rewrite)
```

The agent picks a project, rewrites the question as a search query, chooses a strategy (hybrid, semantic, keyword, symbol lookup, or read whole file), reranks the candidates, checks that the evidence is enough, and verifies that the answer's citations support its claims. If it still can't support an answer after 3 attempts, it returns "Not found".

## Safety

Secrets are kept out of the index ([safety.py](src/kb/safety.py)):
- Files named like `.env`, `*.pem`, `*.key`, `*secret*`, `*credential*`, `id_rsa*` and similar are never read.
- Any file containing an AWS key, private key block, `sk-...` key, password in a URL, or `password = "..."` style assignment is skipped entirely.
- The answer prompt also tells the model never to reveal secrets.

Symlinks and files over 1 MB are skipped.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the repo root:

```
OPENROUTER_API_KEY=your-key-here
# optional, defaults to anthropic/claude-haiku-5.5
OPENROUTER_MODEL=anthropic/claude-haiku-5.5
```

Edit [config.yaml](config.yaml) to choose which projects to index. Every name in `include_projects` must be a top-level folder under `projects_root`, otherwise indexing fails with a clear error.

```yaml
projects_root: ~/Documents
include_projects:
  - my-api
  - my-dashboard
```

Set the `KB_CONFIG` environment variable to use a config file from another location.

The first run downloads the embedding model.

## Usage

Run all commands from the repo root.

```bash
# Build or update the index (only new or changed files are re-embedded)
uv run python -m src.cli index

# Or index every project in a folder, ignoring config.yaml
uv run python -m src.cli index ~/Documents/some-folder

# Ask a question with the full agent loop (shows the steps it took)
uv run python -m src.cli ask "What query do I use to list all chillers?"

# One search and one answer, without the agent loop (faster, cheaper)
uv run python -m src.cli ask "Where is the Influx bucket configured?" --plain
```

Each answer ends with its citations and a summary line with time, number of LLM calls and cost.

### Other scripts

| Script | Purpose |
| --- | --- |
| `uv run python search.py "query"` | Show the top 5 raw retrieval hits, no LLM |
| `uv run python ask.py "question" [--agent]` | Older entry point that also prints token counts |
| `uv run python index.py` | Index using `config.yaml` |
| `uv run python main.py` | Dry run: list discovered files and chunks per project |

## Use it from Claude Code, Cursor or VS Code (MCP)

[mcp_server.py](src/mcp_server.py) exposes the copilot over stdio:

| Tool | What it does |
| --- | --- |
| `ask_codebase` | Answer a question with `file:line` citations |
| `search_code` | Return matching locations and snippets, optionally for one project |
| `list_projects` | List indexed projects and their chunk counts |
| `reindex` | Update the index with the latest changes |

For example, with Claude Code:

```bash
claude mcp add codebase-copilot -- uv --directory /absolute/path/to/Codebase-QnA-copilot run python -m src.mcp_server
```

## Evaluation

[evals/questions.yaml](evals/questions.yaml) holds a draft set of 20 questions (lookups, queries, conceptual, cross-project, plus negative and safety cases). Each lists where its answer lives, so retrieval can be scored with recall@k:

```bash
uv run python -m evals.run_retrieval [run_name] [vector|keyword|hybrid]
```

Results are saved in [evals/results/](evals/results). Recall@5 on the questions that have an expected location, against a 22,710-chunk index:

| Run | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- |
| `baseline` (vector, no path header) | 0.333 | 0.500 | 0.556 |
| `header_vector` | 0.444 | 0.611 | 0.722 |
| `header_keyword` | 0.500 | 0.611 | 0.611 |
| `header_hybrid` | 0.389 | 0.667 | 0.667 |

Adding the project and path header to each embedded chunk gave the biggest improvement. The question set is still a draft, so treat these numbers as directional.

## Project layout

```
config.yaml          which projects to index
src/
  cli.py             index / ask commands
  mcp_server.py      MCP server
  kb/                discovery, chunking, embedding, stores, hybrid retrieval, safety
  rag/               OpenRouter client, prompt, citation handling
  agent/             LangGraph loop, search strategies, LLM decision provider
evals/               question set, retrieval scorer, saved results
chroma_db/           generated index (git-ignored)
```

The index lives in `chroma_db/` (Chroma, `keyword.sqlite3` and `manifest.json`). Delete that folder to rebuild from scratch.
