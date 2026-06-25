"""
File Manager Module

Responsible for:
- Creating directories and writing files safely inside a workspace
- Tracking files created/modified during the session
- Strict path traversal & symlink protection
- Atomic writes with rollback on failure
- Optional size limits and read/delete helpers
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path("generated_projects/current_project")

MAX_FILE_BYTES = 5 * 1024 * 1024            # 5 MB hard cap per file
MAX_PATH_LENGTH = 255                       # POSIX-friendly limit
EXCLUDED_DIRS = {                           # Skipped during listing
    ".git", "__pycache__", "node_modules",
    ".venv", "venv", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".idea", ".vscode",
}

WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Thread-safe tracking of files created/modified this session
_created_files_lock = threading.Lock()
_created_files: Set[str] = set()


# ---------------------------------------------------------------------------
# Workspace bootstrap
# ---------------------------------------------------------------------------

def ensure_workspace() -> Path:
    """Create the workspace directory if missing and return its resolved path."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT.resolve()


def get_workspace_root() -> Path:
    """Public accessor for the resolved workspace root."""
    return ensure_workspace()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _validate_relative_path(file_path: str) -> None:
    """
    Pre-validate the *input* string before any filesystem resolution.

    Raises:
        ValueError: if the path is unsafe.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")

    if not file_path.strip():
        raise ValueError("file_path must be non-empty")

    if len(file_path) > MAX_PATH_LENGTH:
        raise ValueError(f"file_path exceeds {MAX_PATH_LENGTH} chars")

    if "\x00" in file_path:
        raise ValueError("Null byte in file_path")

    if "\\" in file_path:
        raise ValueError(f"Backslashes not allowed in file_path: {file_path!r}")

    if file_path.startswith(("/", "~")):
        raise ValueError(f"Path must be project-relative: {file_path!r}")

    if re.match(r"^[A-Za-z]:", file_path):
        raise ValueError(f"Windows drive letter not allowed: {file_path!r}")

    posix = PurePosixPath(file_path)
    if posix.is_absolute():
        raise ValueError(f"Absolute path not allowed: {file_path!r}")

    if any(part == ".." for part in posix.parts):
        raise ValueError(f"Path traversal detected: {file_path!r}")

    win = PureWindowsPath(file_path)
    if win.is_absolute() or win.drive:
        raise ValueError(f"Windows absolute path detected: {file_path!r}")

    for part in posix.parts:
        stem = part.split(".")[0].lower()
        if stem in WINDOWS_RESERVED:
            raise ValueError(
                f"Reserved Windows filename in path: {file_path!r}"
            )


def _resolve_inside_workspace(file_path: str) -> Path:
    """
    Validate and resolve a project-relative path to an absolute path
    inside the workspace. Raises ValueError on any escape attempt.
    """
    _validate_relative_path(file_path)
    workspace = ensure_workspace()

    candidate = (workspace / file_path).resolve()

    # Strict containment check (works on all Python versions)
    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Path escapes workspace after resolution: {file_path!r}"
        )

    # Symlink defense: ensure no ancestor up to workspace is a symlink
    _reject_symlink_ancestors(candidate, workspace)

    return candidate


def _reject_symlink_ancestors(candidate: Path, workspace: Path) -> None:
    """Walk from candidate up to workspace; reject any symlinked component."""
    current = candidate
    while current != workspace:
        if current.is_symlink():
            raise ValueError(f"Symlinked path component rejected: {current}")
        if current.parent == current:
            break
        current = current.parent


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(target: Path, content: str) -> None:
    """
    Write `content` to `target` atomically:
    write to a temp file in the same directory, fsync, then rename.
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
        os.replace(tmp, target)  # atomic on POSIX & Windows (Py 3.3+)
    except Exception:
        # Best-effort cleanup
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to clean temp file: %s", tmp, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_or_update_file(
    file_path: str,
    file_content: str,
) -> Tuple[bool, str, List[str]]:
    """
    Create or update a file inside the workspace.

    Returns:
        (success, error_message, [relative_path_written])
    """
    logger.info("create_or_update_file: %r (%d chars)",
                file_path, len(file_content) if file_content else 0)

    if not isinstance(file_content, str):
        return False, "file_content must be a string", []

    encoded_size = len(file_content.encode("utf-8"))
    if encoded_size > MAX_FILE_BYTES:
        return (
            False,
            f"file_content exceeds {MAX_FILE_BYTES} bytes (got {encoded_size})",
            [],
        )

    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        logger.warning("Rejected path %r: %s", file_path, exc)
        return False, str(exc), []

    # Skip no-op writes
    if target.exists() and target.is_file():
        try:
            if target.read_text(encoding="utf-8") == file_content:
                rel = _relative(target)
                logger.info("Unchanged, skipping write: %s", rel)
                _track(rel)
                return True, "", [rel]
        except (OSError, UnicodeDecodeError):
            # Fall through to overwrite
            pass

    try:
        _atomic_write(target, file_content)
    except PermissionError as exc:
        msg = f"Permission denied writing {file_path!r}: {exc}"
        logger.error(msg)
        return False, msg, []
    except OSError as exc:
        msg = f"OS error writing {file_path!r}: {exc}"
        logger.error(msg)
        return False, msg, []
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error writing %r", file_path)
        return False, f"Unexpected error: {exc}", []

    if not target.exists():
        msg = f"Write reported success but file missing: {target}"
        logger.error(msg)
        return False, msg, []

    size = target.stat().st_size
    rel = _relative(target)
    logger.info("Wrote %s (%d bytes)", rel, size)
    _track(rel)
    return True, "", [rel]


def verify_file_exists(file_path: str) -> bool:
    """Return True if the (validated) path exists as a file inside the workspace."""
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError:
        return False
    return target.is_file()


def read_file(file_path: str) -> Tuple[bool, str, str]:
    """
    Read a file from the workspace.

    Returns:
        (success, content_or_error, relative_path)
    """
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        return False, str(exc), ""

    if not target.is_file():
        return False, f"File not found: {file_path}", ""

    try:
        content = target.read_text(encoding="utf-8")
        return True, content, _relative(target)
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"Read error: {exc}", ""


def delete_file(file_path: str) -> Tuple[bool, str]:
    """Delete a file inside the workspace."""
    try:
        target = _resolve_inside_workspace(file_path)
    except ValueError as exc:
        return False, str(exc)

    if not target.exists():
        return False, f"File not found: {file_path}"
    if not target.is_file():
        return False, f"Not a regular file: {file_path}"

    try:
        target.unlink()
    except OSError as exc:
        return False, f"Delete failed: {exc}"

    rel = _relative(target)
    with _created_files_lock:
        _created_files.discard(rel)
    logger.info("Deleted %s", rel)
    return True, ""


def list_created_files(tracked_only: bool = False) -> List[str]:
    """
    List files in the workspace, as paths relative to the workspace root.

    Args:
        tracked_only: If True, return only files written via this module
                      in the current session.
    """
    if tracked_only:
        with _created_files_lock:
            return sorted(_created_files)

    workspace = ensure_workspace()
    if not workspace.exists():
        return []

    results: List[str] = []
    for path in workspace.rglob("*"):
        # Skip excluded directories anywhere in the hierarchy
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            results.append(_relative(path))
    return sorted(results)


def reset_tracking() -> None:
    """Clear the in-memory tracker (useful between sessions/tests)."""
    with _created_files_lock:
        _created_files.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _relative(target: Path) -> str:
    """Return a forward-slash, workspace-relative path string."""
    workspace = ensure_workspace()
    return target.relative_to(workspace).as_posix()


def _track(relative_path: str) -> None:
    with _created_files_lock:
        _created_files.add(relative_path)


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
        "src/utils/helpers.py", "def add(a, b):\n    return a + b\n"
    )
    print(f"create nested  -> ok={ok}, err={err!r}, files={files}")

    # Unchanged write (should be a no-op)
    ok, err, files = create_or_update_file("main.py", "print('hello world')\n")
    print(f"rewrite same   -> ok={ok}, err={err!r}, files={files}")

    # Path traversal attempt
    ok, err, files = create_or_update_file("../evil.py", "boom")
    print(f"traversal      -> ok={ok}, err={err!r}")

    # Absolute path attempt
    ok, err, files = create_or_update_file("/etc/passwd", "boom")
    print(f"absolute       -> ok={ok}, err={err!r}")

    # Windows path attempt
    ok, err, files = create_or_update_file("C:\\Windows\\evil.py", "boom")
    print(f"windows abs    -> ok={ok}, err={err!r}")

    # Verify + read
    print(f"\nexists main.py = {verify_file_exists('main.py')}")
    ok, content, rel = read_file("main.py")
    print(f"read main.py   -> ok={ok}, content={content!r}, rel={rel}")

    print("\nAll workspace files:")
    for f in list_created_files():
        print(f"  {f}")

    print("\nTracked this session:")
    for f in list_created_files(tracked_only=True):
        print(f"  {f}")