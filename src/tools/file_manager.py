"""
File Manager

Safe file I/O for the agent's workspace. Handles:
- Path validation (no traversal, no absolute paths, no reserved names)
- Atomic writes via tempfile + rename
- Size limits
- Thread-safe tracking of files written this session

All paths are project-relative. The workspace is the agent's sandbox.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path("generated_projects/current_project")

MAX_FILE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_PATH_LENGTH = 255

# Folders to skip when listing — these regenerate themselves and add noise
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules",
    ".venv", "venv", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".idea", ".vscode",
}

# Reserved on Windows even as part of a filename (case-insensitive)
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Patterns that immediately disqualify a path string
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:")

# Session tracking — thread-safe because Streamlit can rerender concurrently
_tracking_lock = threading.Lock()
_created_files: Set[str] = set()


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def ensure_workspace() -> Path:
    """Create the workspace if missing and return its resolved absolute path."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT.resolve()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _validate_path_string(file_path: str) -> None:
    """
    Reject obviously-bad input before touching the filesystem.

    Raises ValueError with a descriptive message.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string")

    if len(file_path) > MAX_PATH_LENGTH:
        raise ValueError(f"file_path exceeds {MAX_PATH_LENGTH} chars")

    if "\x00" in file_path:
        raise ValueError("null byte in file_path")

    if "\\" in file_path:
        raise ValueError(f"backslashes not allowed: {file_path!r}")

    if file_path.startswith(("/", "~")):
        raise ValueError(f"path must be project-relative: {file_path!r}")

    if _DRIVE_LETTER_RE.match(file_path):
        raise ValueError(f"drive letter not allowed: {file_path!r}")

    parts = PurePosixPath(file_path).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"path traversal detected: {file_path!r}")

    for part in parts:
        stem = part.split(".")[0].lower()
        if stem in WINDOWS_RESERVED:
            raise ValueError(f"reserved filename in path: {file_path!r}")


def _resolve_inside_workspace(file_path: str) -> Path:
    """
    Validate the input, then resolve to an absolute path that we've
    confirmed lives inside the workspace.
    """
    _validate_path_string(file_path)
    workspace = ensure_workspace()
    candidate = (workspace / file_path).resolve()

    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise ValueError(f"path escapes workspace: {file_path!r}")

    if candidate.is_symlink():
        raise ValueError(f"symlinked path rejected: {candidate}")

    return candidate


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(target: Path, content: str) -> None:
    """
    Write content to a temp file in the same directory, fsync, then rename.

    os.replace is atomic on POSIX and Windows (Python 3.3+), so the file
    either contains the full new content or the previous version — never
    a partial write.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp = Path(tmp_path)

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_or_update_file(
    file_path: str,
    file_content: str,
) -> Tuple[bool, str, List[str]]:
    """
    Write a file inside the workspace.

    Returns (success, error_message, [relative_path_written]).
    The error_message is empty on success.
    """
    if not isinstance(file_content, str):
        return False, "file_content must be a string", []

    encoded_size = len(file_content.encode("utf-8"))
    if encoded_size > MAX_FILE_BYTES:
        return False, f"file exceeds {MAX_FILE_BYTES} bytes (got {encoded_size})", []

    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        logger.warning("Rejected path %r: %s", file_path, exc)
        return False, str(exc), []

    # Skip the write if content is already identical — common when the
    # agent revisits the same step.
    if target.is_file() and _content_matches(target, file_content):
        rel = _relative(target)
        _track(rel)
        logger.debug("No change for %s; skipping write", rel)
        return True, "", [rel]

    try:
        _atomic_write(target, file_content)
    except (PermissionError, OSError) as exc:
        msg = f"write failed for {file_path!r}: {exc}"
        logger.error(msg)
        return False, msg, []

    rel = _relative(target)
    _track(rel)
    logger.info("Wrote %s (%d bytes)", rel, encoded_size)
    return True, "", [rel]


def verify_file_exists(file_path: str) -> bool:
    """True if the path exists as a regular file inside the workspace."""
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError:
        return False
    return target.is_file()


def read_file(file_path: str) -> Tuple[bool, str, str]:
    """
    Read a file from the workspace.

    Returns (success, content_or_error, relative_path).
    """
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        return False, str(exc), ""

    if not target.is_file():
        return False, f"file not found: {file_path}", ""

    try:
        return True, target.read_text(encoding="utf-8"), _relative(target)
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"read failed: {exc}", ""


def delete_file(file_path: str) -> Tuple[bool, str]:
    """Delete a file inside the workspace. Returns (success, error_message)."""
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        return False, str(exc)

    if not target.is_file():
        return False, f"file not found or not regular: {file_path}"

    try:
        target.unlink()
    except OSError as exc:
        return False, f"delete failed: {exc}"

    rel = _relative(target)
    with _tracking_lock:
        _created_files.discard(rel)
    logger.info("Deleted %s", rel)
    return True, ""


def list_created_files(tracked_only: bool = False) -> List[str]:
    """
    List files in the workspace, as workspace-relative paths.

    tracked_only=True returns only files written via this module
    in the current session (useful for "what did the agent produce?").
    """
    if tracked_only:
        with _tracking_lock:
            return sorted(_created_files)

    workspace = ensure_workspace()
    if not workspace.exists():
        return []

    results = []
    for path in workspace.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            results.append(_relative(path))
    return sorted(results)


def reset_tracking() -> None:
    """Clear the in-memory tracker — useful for tests or between sessions."""
    with _tracking_lock:
        _created_files.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative(target: Path) -> str:
    """Workspace-relative path string with forward slashes."""
    return target.relative_to(ensure_workspace()).as_posix()


def _track(relative_path: str) -> None:
    with _tracking_lock:
        _created_files.add(relative_path)


def _content_matches(target: Path, content: str) -> bool:
    """True if the file's current content equals `content`."""
    try:
        return target.read_text(encoding="utf-8") == content
    except (OSError, UnicodeDecodeError):
        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing File Manager...\n")

    # Happy path
    ok, err, files = create_or_update_file("main.py", "print('hello world')\n")
    print(f"create main.py -> ok={ok}, err={err!r}, files={files}")

    # Nested path
    ok, err, files = create_or_update_file(
        "src/utils/helpers.py", "def add(a, b):\n    return a + b\n",
    )
    print(f"create nested  -> ok={ok}, err={err!r}, files={files}")

    # Idempotent rewrite (no-op)
    ok, err, files = create_or_update_file("main.py", "print('hello world')\n")
    print(f"rewrite same   -> ok={ok}, err={err!r}, files={files}")

    # Rejection cases
    for bad_path in ("../evil.py", "/etc/passwd", "C:\\Windows\\evil.py"):
        ok, err, _ = create_or_update_file(bad_path, "boom")
        print(f"reject {bad_path!r:25} -> ok={ok}, err={err}")

    # Read back
    print(f"\nexists main.py = {verify_file_exists('main.py')}")
    ok, content, rel = read_file("main.py")
    print(f"read main.py   -> ok={ok}, content={content!r}")

    print("\nAll workspace files:")
    for f in list_created_files():
        print(f"  {f}")

    print("\nTracked this session:")
    for f in list_created_files(tracked_only=True):
        print(f"  {f}")