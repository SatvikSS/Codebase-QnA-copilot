"""MCP server, so Claude Code, Cursor and VS Code can use the copilot.

Run from the repo root:  uv run python -m src.mcp_server
Heavy imports happen inside the tools, so the server starts instantly.
Nothing may print to stdout here: stdio MCP uses it for the protocol.
"""

from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("codebase-copilot")

SNIPPET_CHARS = 600
MAX_RESULTS = 20


@mcp.tool()
def ask_codebase(question: str) -> str:
    """Answer a question about the user's own indexed projects (code, queries, configs, docs).
    The answer cites the exact file and line range it came from."""
    from src.agent.graph import run_agent
    from src.rag.answer import with_citations

    return with_citations(run_agent(question).answer)


@mcp.tool()
def search_code(query: str, project: str | None = None, limit: int = 5) -> list[dict]:
    """Search the indexed projects without writing an answer. Returns matching file locations
    and snippets. Pass project to search only one project (see list_projects)."""
    from src.kb.retrieval import search_chunks

    hits = search_chunks(query, k=max(1, min(limit, MAX_RESULTS)), project=project)
    return [
        {
            "project": hit["project"],
            "file": hit["path"],
            "start_line": hit["start_line"],
            "end_line": hit["end_line"],
            "snippet": hit["text"][:SNIPPET_CHARS],
        }
        for hit in hits
    ]


@mcp.tool()
def list_projects() -> list[dict]:
    """List the projects that are included in the index, with how many chunks each has."""
    from src.kb.config import load_config
    from src.kb.keyword import chunks_per_project

    counts = chunks_per_project()
    return [
        {"project": name, "chunks": counts.get(name, 0)}
        for name in sorted(load_config().include_projects)
    ]


@mcp.tool()
def reindex() -> dict:
    """Update the index with the latest changes to the projects. Only new, edited or deleted
    files are processed, so this is quick after the first run."""
    from src.kb.config import load_config
    from src.kb.indexer import index_projects

    config = load_config()
    return asdict(index_projects(config.projects_root, config.include_projects))


if __name__ == "__main__":
    mcp.run()
