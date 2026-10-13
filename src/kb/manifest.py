"""Remember each indexed file's hash, so only changed files are re-indexed."""

import hashlib
import json

from .discover import SourceFile
from .store import DB_PATH

MANIFEST_PATH = DB_PATH / "manifest.json"


def file_key(file: SourceFile) -> str:
    return f"{file.project}/{file.rel_path}"


def file_hash(file: SourceFile) -> str:
    return hashlib.sha256(file.path.read_bytes()).hexdigest()


def load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True))
