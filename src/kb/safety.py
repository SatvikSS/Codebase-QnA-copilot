"""Keep secrets out of the index."""

import re
from fnmatch import fnmatch
from pathlib import Path

BLOCKED_FILE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.crt", "*.p12", "*.pfx",
    "id_rsa*", "id_ed25519*", ".netrc", "*credential*", "*secret*",
    "agents.md", "claude.md", ".ds_store", ".gitignore", "*.pkl"
    "*.svg", "*.png", "*.jpg", "*.webp", ".dockerignore",
)

SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),                                   # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),                 # private key block
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),                              # OpenAI/OpenRouter style key
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),                             # password inside a URL
    re.compile(                                                        # password = "value"
        r"""(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['"][^'"\s]{8,}['"]""",
        re.IGNORECASE,
    ),
)


def is_sensitive_path(path: Path) -> bool:
    """True if the file name looks like a secrets file (checked before reading it)."""
    name = path.name.lower()
    return any(fnmatch(name, pattern) for pattern in BLOCKED_FILE_PATTERNS)


def contains_secret(text: str) -> bool:
    """True if the text matches a known secret pattern (checked before embedding it)."""
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)
