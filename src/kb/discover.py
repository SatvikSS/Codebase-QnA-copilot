"""Find indexable source files under a projects directory."""

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .safety import is_sensitive_path


SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".idea", ".worktrees",
    ".claude", ".cursor", ".codex", ".kiro",
    ".vscode", ".beads",
})


@dataclass(frozen=True)
class SourceFile:
    project: str
    path: Path          # absolute
    rel_path: str       # relative to project root, always "/" separated
    size: int
    modified_at: float


def list_projects(root: Path, include: frozenset[str] | None = None) -> list[Path]:
    """Top-level, non-hidden directories of root, sorted.

    If include is given, keep only those names and raise if any are missing,
    so a typo in config.yaml fails loudly instead of silently skipping a project.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Projects root not found: {root}")

    projects = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if include is None:
        return projects

    missing = include - {p.name for p in projects}
    if missing:
        raise ValueError(f"Projects not found under {root}: {sorted(missing)}")
    return [p for p in projects if p.name in include]


def git_listed_files(repo: Path) -> list[Path]:
    """Files git considers part of the repo (tracked + untracked, minus ignored).

    A nested repo shows up as a single directory entry like "inner/".
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return [repo / rel for rel in result.stdout.split("\0") if rel]


def should_include(path: Path, project: Path) -> bool:
    """Skip files inside SKIP_DIRS, symlinks, and anything that is not a regular file."""
    if any(part in SKIP_DIRS for part in path.relative_to(project).parts[:-1]):
        return False
    if is_sensitive_path(path):
        return False
    # Symlinks could point outside the project (e.g. to ~), so skip them
    return path.is_file() and not path.is_symlink()


def candidate_files(folder: Path) -> Iterator[Path]:
    """Files under folder. Uses git where a .git exists, otherwise walks the folder.

    Repos can sit inside plain folders or inside other repos, so this recurses.
    """
    if (folder / ".git").exists():
        for path in git_listed_files(folder):
            if path.is_dir() and not path.is_symlink():  # a nested repo
                yield from candidate_files(path)
            else:
                yield path
        return

    for child in sorted(folder.iterdir()):
        if child.is_symlink():
            continue
        if child.is_dir():
            if child.name not in SKIP_DIRS:
                yield from candidate_files(child)
        else:
            yield child


def discover(root: Path, include: frozenset[str] | None = None) -> Iterator[SourceFile]:
    for project in list_projects(root.resolve(), include):
        for path in candidate_files(project):
            if not should_include(path, project):
                continue
            stat = path.stat()
            yield SourceFile(
                project=project.name,
                path=path,
                rel_path=path.relative_to(project).as_posix(),
                size=stat.st_size,
                modified_at=stat.st_mtime,
            )
