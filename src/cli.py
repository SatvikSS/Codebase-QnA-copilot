"""Command line: index your projects and ask questions about them.

Run from the repo root:
    uv run python -m src.cli index [PROJECTS_DIR]
    uv run python -m src.cli ask "your question" [--plain]
"""

import time
from pathlib import Path

import typer

app = typer.Typer(help="Codebase Q&A copilot", no_args_is_help=True, add_completion=False)


@app.command()
def index(
    projects_dir: Path | None = typer.Argument(
        None,
        help="Folder with your projects. Without it, projects_root and include_projects from config.yaml are used.",
    ),
) -> None:
    """Index your projects. Only new or changed files are re-embedded."""
    from src.kb.config import load_config
    from src.kb.indexer import index_projects

    config = load_config()
    if projects_dir is None:
        stats = index_projects(config.projects_root, config.include_projects)
    else:
        # An explicit folder means: index every project in it
        stats = index_projects(projects_dir.expanduser().resolve(), None)

    typer.echo(
        f"files: {stats.files} | new or changed: {stats.changed} | deleted: {stats.deleted}\n"
        f"new chunks: {stats.new_chunks} | chunks in database: {stats.total_chunks} | {stats.seconds}s"
    )


@app.command()
def ask(
    question: str,
    plain: bool = typer.Option(False, "--plain", help="One search and one answer, without the agent loop."),
) -> None:
    """Ask a question about your projects. The answer cites file:line."""
    from src.agent.graph import run_agent
    from src.rag.answer import answer_question, with_citations

    start = time.perf_counter()
    if plain:
        answer = answer_question(question)
        responses = (answer.response,)
    else:
        result = run_agent(question)
        answer, responses = result.answer, result.responses
        typer.echo("Agent steps:")
        for step in result.trace:
            typer.echo(f"  {step}")
        typer.echo()
    elapsed = time.perf_counter() - start

    typer.echo(with_citations(answer))
    cost = sum(r.cost or 0 for r in responses)
    typer.echo(f"\n{elapsed:.1f}s | {len(responses)} LLM calls | cost ${cost:.5f}")


if __name__ == "__main__":
    app()
