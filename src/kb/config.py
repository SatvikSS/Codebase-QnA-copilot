"""Load and validate config.yaml."""

from dataclasses import dataclass
from pathlib import Path
import os
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(os.environ.get("KB_CONFIG", REPO_ROOT / "config.yaml"))


@dataclass(frozen=True)
class Config:
    projects_root: Path
    include_projects: frozenset[str]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read config.yaml and return a Config object."""

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    root = raw.get("projects_root")
    include = raw.get("include_projects")

    return Config(
        projects_root=Path(root).expanduser().resolve(),
        include_projects=frozenset(name.strip() for name in include),
    )