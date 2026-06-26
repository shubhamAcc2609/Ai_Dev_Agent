"""
Execution Manager

Runs shell commands safely inside the agent's workspace.

Safety layers (in order of application):
    1. Validation — command must pass allow-list + dangerous-pattern checks
    2. Normalization — translate Linux idioms to Windows on Windows hosts
    3. Sandboxed execution — cwd locked to workspace, output captured, timeout
       enforced with cross-platform process-tree kill

Note: the library_manager layer prevents the planner from generating commands
that depend on tools missing from the host. This module's allow-list is the
defense-in-depth backstop, not the primary gate.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"
WORKSPACE_ROOT = Path("generated_projects/current_project")

MAX_OUTPUT_BYTES = 1_000_000
DEFAULT_TIMEOUT_SEC = 30

# Curated allow-list of tools the agent may invoke.
# Anything else needs to either live in the workspace (e.g. compiled binary)
# or be added via the AGENT_ALLOWED_BINARIES environment variable.
_BUILTIN_ALLOWED = {
    # Python
    "python", "python3", "py", "pip", "pip3", "pytest",
    "uvicorn", "gunicorn", "flask", "streamlit",
    # Node
    "node", "npm", "npx", "yarn", "pnpm",
    # Compiled languages
    "gcc", "g++", "clang", "rustc", "cargo", "go",
    "javac", "java", "dotnet",
    # Build & VCS
    "make", "cmake", "git",
    # Shell basics
    "echo", "ls", "dir", "cat", "type",
    "mkdir", "cp", "copy", "mv", "move", "rm", "del",
    # Network
    "curl", "wget",
    # Lint / format
    "ruff", "black", "mypy", "eslint", "prettier",
}


def _load_allow_list() -> Set[str]:
    """Combine the built-in list with any AGENT_ALLOWED_BINARIES env entries."""
    allowed = set(_BUILTIN_ALLOWED)
    extra = os.environ.get("AGENT_ALLOWED_BINARIES", "").strip()
    if extra:
        for token in re.split(r"[,\s]+", extra):
            token = token.strip().lower()
            if token:
                allowed.add(token)
    return allowed


ALLOWED_BINARIES = _load_allow_list()


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def ensure_workspace() -> Path:
    """Create the workspace if missing; return its resolved path."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT.resolve()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_workspace_binary(token: str, workspace: Path) -> bool:
    """True if `token` refers to an executable file living in the workspace."""
    if not token or ".." in token:
        return False

    cleaned = token.lstrip("./\\")
    if not cleaned:
        return False

    # Try the name as-is and with .exe / .out — covers all common compiled outputs.
    for name in (cleaned, cleaned + ".exe", cleaned + ".out"):
        try:
            candidate = (workspace / name).resolve()
        except (OSError, ValueError):
            continue
        if _is_inside(candidate, workspace) and candidate.is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Normalization (Linux idioms → Windows equivalents)
# ---------------------------------------------------------------------------

_WINDOWS_REPLACEMENTS = [
    (re.compile(r"\bpython3\b"), "python"),
    (re.compile(r"\bpip3\b"), "pip"),
    (re.compile(r"\bls\s+-l[a]?\b"), "dir"),
    (re.compile(r"\bls\b"), "dir"),
    (re.compile(r"\bcat\b"), "type"),
    (re.compile(r"\bcp\b"), "copy"),
    (re.compile(r"\bmv\b"), "move"),
    (re.compile(r"\brm\b"), "del"),
    (re.compile(r"(?<!\S)\./(?=\S)"), r".\\"),   # ./prog → .\prog
]


def normalize_command(command: str) -> str:
    """Translate Linux-style commands to Windows equivalents (no-op on POSIX)."""
    if not IS_WINDOWS:
        return command
    for pattern, replacement in _WINDOWS_REPLACEMENTS:
        command = pattern.sub(replacement, command)
    return command


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Shell metacharacters that enable real injection (we allow && / || / ; but
# handle them by splitting sub-commands, not by passing them through).
FORBIDDEN_METACHARS = ("`", "$(", ">", "<", "|", "\n", "\r")

SEQUENCE_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")

DANGEROUS_PATTERNS = (
    re.compile(r"\brm\s+-[rf]+\s+[/~*]", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fsq]+(\s+/[fsq]+)*\s+[a-z]:\\", re.IGNORECASE),
    re.compile(r"\b(?:mkfs|shutdown|reboot|sudo)\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*:\|:&\s*\};:"),              # fork bomb
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
)

# Recognizes binaries produced by `-o NAME` in a compile sub-command, so
# `g++ swap.cpp -o swap && ./swap` validates even when `swap` doesn't exist yet.
COMPILE_OUTPUT_RE = re.compile(
    r"\b(?:gcc|g\+\+|clang|clang\+\+|rustc|go\s+build)\b[^&|;]*\s-o\s+(\S+)",
    re.IGNORECASE,
)


def _extract_compile_outputs(command: str) -> Set[str]:
    """Find binary names that compile steps in this command will produce."""
    outputs = set()
    for match in COMPILE_OUTPUT_RE.finditer(command):
        name = match.group(1).lstrip("./\\")
        for ext in (".exe", ".out"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        if name:
            outputs.add(name.lower())
    return outputs


def validate_command(command: str) -> Tuple[bool, str]:
    """
    Validate a command before execution.

    Allows && / || / ; sequencing but validates each sub-command independently.
    Returns (is_valid, error_message).
    """
    if not isinstance(command, str):
        return False, "command must be a string"

    stripped = command.strip()
    if not stripped:
        return False, "command cannot be empty"
    if len(stripped) > 1000:
        return False, "command too long (>1000 chars)"

    sub_commands = [s for s in SEQUENCE_SPLIT_RE.split(stripped) if s.strip()]
    if not sub_commands:
        return False, "no executable found"

    workspace = ensure_workspace()
    expected_outputs = _extract_compile_outputs(stripped)

    for sub in sub_commands:
        for tok in FORBIDDEN_METACHARS:
            if tok in sub:
                return False, f"disallowed metacharacter {tok!r} in: {sub!r}"

        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(sub):
                return False, f"dangerous command blocked: {sub!r}"

        try:
            tokens = shlex.split(sub, posix=not IS_WINDOWS)
        except ValueError as exc:
            return False, f"unparseable sub-command {sub!r}: {exc}"
        if not tokens:
            return False, f"empty sub-command in: {command!r}"

        first = tokens[0]
        binary = re.sub(
            r"\.(exe|bat|cmd|out|sh|py)$", "",
            os.path.basename(first).lower(),
        )

        # cd is fine within a sequence — just directory navigation
        if binary == "cd":
            continue

        # `python -m module ...` is trusted (python itself is allow-listed)
        if binary in ("python", "python3", "py") and len(tokens) >= 3 \
                and tokens[1] == "-m":
            continue

        # Anything in the workspace (compiled binaries, scripts the agent wrote)
        if _is_workspace_binary(first, workspace):
            continue

        # Outputs from an earlier compile sub-command in this same sequence
        if binary in expected_outputs:
            continue

        if binary not in ALLOWED_BINARIES:
            return False, (
                f"binary {binary!r} not in allow-list. "
                f"Add to AGENT_ALLOWED_BINARIES env var, or build it inside "
                f"the workspace before invoking."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _truncate(stream: str) -> str:
    """Cap a stream at MAX_OUTPUT_BYTES with a truncation marker."""
    if len(stream) <= MAX_OUTPUT_BYTES:
        return stream
    return stream[:MAX_OUTPUT_BYTES] + "\n...[output truncated]"


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Cross-platform termination of a process and its children."""
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as exc:
        logger.debug("Process cleanup skipped: %s", exc)


def execute_command(
    command: str,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Run a shell command inside the workspace.

    Returns (success, stdout, stderr). On timeout, stderr includes a notice
    and the process tree is killed.
    """
    ok, msg = validate_command(command)
    if not ok:
        logger.warning("Validation failed: %s", msg)
        return False, "", msg

    command = normalize_command(command.strip())
    workspace = ensure_workspace()

    if cwd is None:
        cwd_path = workspace
    else:
        cwd_path = Path(cwd).resolve()
        if not _is_inside(cwd_path, workspace):
            return False, "", f"cwd must be inside workspace: {workspace}"
        if not cwd_path.exists():
            return False, "", f"cwd does not exist: {cwd_path}"

    popen_kwargs = dict(
        shell=True,
        cwd=str(cwd_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    logger.info("Running: %r (timeout=%ds)", command, timeout)

    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except (FileNotFoundError, OSError) as exc:
        logger.error("Launch failed: %s", exc)
        return False, "", f"failed to launch: {exc}"

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Timeout after %ds; killing process tree", timeout)
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return (
            False,
            _truncate(stdout or ""),
            _truncate(stderr or "") + f"\nCommand timed out after {timeout}s",
        )

    stdout = _truncate(stdout or "")
    stderr = _truncate(stderr or "")
    success = process.returncode == 0

    logger.info(
        "Command %s (rc=%s)",
        "succeeded" if success else "failed",
        process.returncode,
    )
    return success, stdout, stderr