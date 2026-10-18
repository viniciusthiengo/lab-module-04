"""File paths sent by clients: clean, relative, without ``..``."""

import re

MAX_PATH_CHARS = 300
_SAFE_PATH = re.compile(r"^[\w@+\-. /]+$")


class InvalidPathError(ValueError):
    """A path that is not a safe, relative file path."""


def normalize_path(raw: str) -> str:
    """``raw`` as a clean relative path (``src/app.py``); raises ``InvalidPathError``."""
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not path:
        raise InvalidPathError("A file path is empty.")
    if len(path) > MAX_PATH_CHARS:
        raise InvalidPathError(f"The path '{path[:40]}…' is too long.")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise InvalidPathError(f"The path '{path}' must be relative.")
    if not _SAFE_PATH.match(path):
        raise InvalidPathError(f"The path '{path}' has characters that are not allowed.")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise InvalidPathError(f"The path '{path}' must not have empty, '.' or '..' parts.")
    return path
