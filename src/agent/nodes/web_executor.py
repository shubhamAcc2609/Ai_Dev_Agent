"""
Web Executor

Handles web applications: FastAPI, Flask, Streamlit, Django, plain HTTP servers.

Characteristics:
- Three command modes, each with specialized handling:
    1. install:  long timeout (180s) for pip/npm dependency installation
    2. server:   background launch + HTTP endpoint probing for verification
    3. regular:  standard timeout for everything else (e.g. pytest, scripts)
- HTTP probe is the gold-standard verification — actually proves the app works

Bulletproof guarantees:
- Never raises — all exceptions caught and translated to failure deltas
- Defensive against malformed state (handled by shared helpers)
- Defensive against malformed code-generator output (validated locally)
- Defensive against execute_command crashes (caught and wrapped)
- Defensive against subprocess / network failures during probing
- Cross-platform process-tree cleanup (Windows taskkill / POSIX killpg)
- Returns valid state delta in every code path

Specialization from simple/compiled executors:
- Recognizes install commands via _is_install()
- Recognizes server commands via _is_server() pattern matching (install priority)
- Background server launch via _run_server_and_probe()
- HTTP probing with 2xx-prefer / 4xx-accept-routing strategy
- Port extraction from command (--port, :PORT, http.server N, default 8000)
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from src.agent.nodes._executor_shared import (
    STATUS_FAILED,
    StepResult,
    build_success_delta,
    check_all_steps_done,
    check_empty_plan,
    extract_state_basics,
    generate_plan_for_step,
    handle_step_failure,
    make_delta,
    make_logger,
    write_file_if_needed,
)
from src.agent.state import AgentState
from src.tools.execution_manager import (
    IS_WINDOWS,
    ensure_workspace,
    execute_command,
    normalize_command,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# This executor's specific tuning knobs
# ---------------------------------------------------------------------------

INSTALL_TIMEOUT = 180            # pip / npm install can take 2+ minutes
DEFAULT_CMD_TIMEOUT = 30         # Anything else: tests, scripts, etc.

# Server probe configuration
SERVER_DEFAULT_PORT = 8000
SERVER_STARTUP_WAIT_SEC = 10     # How long to wait for "listening on..."
SERVER_PROBE_TIMEOUT_SEC = 3.0   # Per-request HTTP probe timeout
SERVER_PROBE_INTERVAL_SEC = 0.5  # Time between probe attempts
SERVER_PROBE_PATHS = (
    "/docs",            # FastAPI Swagger UI (best signal — 200 if app loaded)
    "/health",          # Common health endpoint
    "/healthz",         # k8s-style health endpoint
    "/",                # Root — last resort
    "/openapi.json",    # FastAPI OpenAPI schema
)

MAX_OUTPUT_PREVIEW = 200         # Chars of stdout/stderr to log per command

# Operations recognized in code-generator output
OP_CREATE_FILE = "create_file"
OP_UPDATE_FILE = "update_file"
OP_RUN_COMMAND = "execute_command"
OP_VERIFY = "verify"
VALID_OPERATIONS = {OP_CREATE_FILE, OP_UPDATE_FILE, OP_RUN_COMMAND, OP_VERIFY}

# Command-prefix patterns that identify package-install commands
INSTALL_PREFIXES = (
    "pip install", "pip3 install", "python -m pip install",
    "npm install", "npm i ", "yarn ", "yarn install",
    "pnpm install", "pnpm i ", "uv pip install",
)

# Regex patterns that identify server-launch commands.
# Each requires the server name be followed by whitespace or end-of-line
# (so `./uvicorn-config-generator` doesn't match, but `uvicorn main:app` does).
SERVER_PATTERNS = (
    re.compile(r"\buvicorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bgunicorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bhypercorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bdaphne(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bstreamlit\s+run\b", re.IGNORECASE),
    re.compile(r"\bpython\s+-m\s+http\.server\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def web_executor_node(state: AgentState) -> dict:
    """
    Execute a single step for a web-application project.

    NEVER RAISES. Every failure mode produces a valid state delta.
    """
    new_logs: List[str] = []
    log = make_logger(new_logs)
    log("--- WEB EXECUTOR ---")

    # ─── Extract state defensively ──────────────────────────────────────
    try:
        s = extract_state_basics(state)
    except Exception as exc:
        logger.exception("State extraction failed catastrophically")
        log(f"CRITICAL: state extraction failed: {exc}", level=logging.ERROR)
        return make_delta(
            logs=new_logs,
            is_complete=True,
            final_status=STATUS_FAILED,
            last_error=f"State extraction failed: {exc}",
        )

    # ─── Terminal-condition guards ──────────────────────────────────────
    delta = check_empty_plan(s, new_logs, log)
    if delta is not None:
        return delta

    delta = check_all_steps_done(s, new_logs, log)
    if delta is not None:
        return delta

    # ─── Execute the current step ───────────────────────────────────────
    step_description = s["plan"][s["current_step"]]
    log(f"Step {s['current_step'] + 1}/{len(s['plan'])}: {step_description}")

    try:
        result = _run_web_step(
            step_description=step_description,
            requirement=s["requirement"],
            plan=s["plan"],
            files_so_far=s["existing_files"],
            log=log,
        )
    except Exception as exc:
        logger.exception("Web executor crashed during step execution")
        log(f"CRITICAL ERROR: {exc}", level=logging.ERROR)
        return handle_step_failure(
            base_logs=s["existing_logs"] + new_logs,
            base_files=s["existing_files"],
            retry_count=s["retry_count"],
            current_step=s["current_step"],
            total_steps=len(s["plan"]),
            step_description=step_description,
            error_msg=str(exc),
            stdout="",
            stderr=str(exc),
            command="<executor crashed>",
            file_path="",
            file_content="",
        )

    # ─── Defensive guard: result must be a StepResult ───────────────────
    if not isinstance(result, StepResult):
        msg = (
            f"_run_web_step returned {type(result).__name__} "
            f"instead of StepResult"
        )
        logger.error(msg)
        log(f"CRITICAL: {msg}", level=logging.ERROR)
        return handle_step_failure(
            base_logs=s["existing_logs"] + new_logs,
            base_files=s["existing_files"],
            retry_count=s["retry_count"],
            current_step=s["current_step"],
            total_steps=len(s["plan"]),
            step_description=step_description,
            error_msg=msg,
            stdout="",
            stderr=msg,
            command="<returned wrong type>",
            file_path="",
            file_content="",
        )

    # ─── Branch on success or failure ───────────────────────────────────
    if result.success:
        return build_success_delta(s, result, new_logs, log)

    return handle_step_failure(
        base_logs=s["existing_logs"] + new_logs,
        base_files=s["existing_files"] + result.files_created,
        retry_count=s["retry_count"],
        current_step=s["current_step"],
        total_steps=len(s["plan"]),
        step_description=step_description,
        error_msg=result.error_message,
        stdout=result.stdout,
        stderr=result.stderr,
        command=result.command,
        file_path=result.file_path,
        file_content=result.file_content,
    )


# ---------------------------------------------------------------------------
# Specialized step runner — what makes this executor "web"
# ---------------------------------------------------------------------------

def _run_web_step(
    *,
    step_description: str,
    requirement: str,
    plan: List[str],
    files_so_far: List[str],
    log,
) -> StepResult:
    """
    Run a single web-project step.

    Three command modes:
      1. install — long timeout (180s) for pip/npm
      2. server — background launch + HTTP probe verification
      3. regular — standard 30s timeout

    GUARANTEE: Always returns a StepResult, never raises.
    """
    result = StepResult()

    # ─── Step 1: Generate the execution plan via code-generator ─────────
    log("[plan] generating execution plan...")
    try:
        plan_dict = generate_plan_for_step(
            step_description, requirement, plan, files_so_far,
        )
    except Exception as exc:
        log(f"[plan] ✗ code-generator failed: {exc}", level=logging.ERROR)
        result.error_message = f"Code generator failed: {exc}"
        result.stderr = str(exc)
        return result

    if not isinstance(plan_dict, dict):
        msg = f"Code generator returned {type(plan_dict).__name__}, expected dict"
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    operation = _safe_get_str(plan_dict, "operation")
    file_path = _safe_get_str(plan_dict, "file_path")
    file_content = _safe_get_str(plan_dict, "file_content", default="")
    command = _safe_get_str(plan_dict, "command")

    result.plan = plan_dict
    result.command = command
    result.file_path = file_path
    result.file_content = file_content

    log(
        f"[plan] operation='{operation}', file_path='{file_path or 'none'}', "
        f"has_command={bool(command)}"
    )

    # ─── Step 2: Operation-specific handling ────────────────────────────

    if operation == OP_VERIFY:
        log("[verify] no-op step (logical checkpoint)")
        result.success = True
        result.verification_summary = "verify step (no action required)"
        return result

    if operation and operation not in VALID_OPERATIONS:
        msg = (
            f"Unknown operation '{operation}'; expected one of "
            f"{sorted(VALID_OPERATIONS)}"
        )
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    # ─── Step 3: Write the file (if requested) ──────────────────────────
    try:
        if not write_file_if_needed(file_path, file_content, result, log):
            return result
    except Exception as exc:
        log(f"[file] ✗ file write crashed: {exc}", level=logging.ERROR)
        result.error_message = f"File write crashed: {exc}"
        result.stderr = str(exc)
        return result

    # ─── Step 4: No command? Step is done ────────────────────────────────
    if not command:
        log("[run] skipped (no command)")
        result.success = True
        return result

    # ─── Step 5: Dispatch to the right command mode ─────────────────────
    # Order matters: install check goes first so `pip install uvicorn`
    # isn't misclassified as a server launch.
    if _is_install(command):
        return _handle_install_command(command, result, log)

    if _is_server(command):
        return _handle_server_command(command, result, log)

    return _handle_regular_command(command, result, log)


# ---------------------------------------------------------------------------
# Mode 1: Server launch + HTTP probe
# ---------------------------------------------------------------------------

def _handle_server_command(command: str, result: StepResult, log) -> StepResult:
    """
    Launch a server in the background, probe its HTTP endpoint, terminate.

    This is the gold-standard verification: we don't just check that the
    process started — we hit an HTTP endpoint and confirm it responds.
    """
    log(f"[server] launching and probing — {command}")
    try:
        ok, stdout, stderr, summary = _run_server_and_probe(command, log)
    except Exception as exc:
        log(f"[server] ✗ probe crashed: {exc}", level=logging.ERROR)
        result.error_message = f"Server probe crashed: {exc}"
        result.stderr = str(exc)
        return result

    result.stdout = stdout
    result.stderr = stderr

    if ok:
        log(f"[server] ✓ verified live: {summary}")
        result.success = True
        result.verification_summary = summary
        return result

    log(
        f"[server] ✗ verification failed ({summary}); "
        f"stderr[:200]={stderr.strip()[:200]!r}",
        level=logging.ERROR,
    )
    result.error_message = (
        f"Server verification failed ({summary}): {stderr.strip()[:200]}"
    )
    return result


# ---------------------------------------------------------------------------
# Mode 2: Install commands (pip, npm, etc.)
# ---------------------------------------------------------------------------

def _handle_install_command(command: str, result: StepResult, log) -> StepResult:
    """Run an install command with the long install timeout."""
    log(f"[install] {command}  (timeout={INSTALL_TIMEOUT}s)")
    try:
        ok, stdout, stderr = execute_command(command, timeout=INSTALL_TIMEOUT)
    except Exception as exc:
        log(f"[install] ✗ execute_command crashed: {exc}", level=logging.ERROR)
        result.error_message = f"Install command crashed: {exc}"
        result.stderr = str(exc)
        return result

    result.stdout = stdout or ""
    result.stderr = stderr or ""

    if not ok:
        preview = (stderr or stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
        log(f"[install] ✗ failed: {preview!r}", level=logging.ERROR)
        result.error_message = f"Install failed: {preview}"
        return result

    stdout_preview = (stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
    log(f"[install] ✓ done. stdout[:120]={stdout_preview[:120]!r}")
    result.verification_summary = (
        f"installed OK; stdout[:80]={stdout_preview[:80]!r}"
    )
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Mode 3: Regular commands (tests, scripts, etc.)
# ---------------------------------------------------------------------------

def _handle_regular_command(command: str, result: StepResult, log) -> StepResult:
    """Run a non-install, non-server command with standard timeout."""
    log(f"[run] {command}  (timeout={DEFAULT_CMD_TIMEOUT}s)")
    try:
        ok, stdout, stderr = execute_command(command, timeout=DEFAULT_CMD_TIMEOUT)
    except Exception as exc:
        log(f"[run] ✗ execute_command crashed: {exc}", level=logging.ERROR)
        result.error_message = f"Command crashed: {exc}"
        result.stderr = str(exc)
        return result

    result.stdout = stdout or ""
    result.stderr = stderr or ""

    if not ok:
        preview = (stderr or stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
        log(f"[run] ✗ failed: {preview!r}", level=logging.ERROR)
        result.error_message = f"Command failed: {preview}"
        return result

    stdout_preview = (stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
    log(f"[run] ✓ exit=0  stdout[:120]={stdout_preview[:120]!r}")
    result.verification_summary = f"exit=0; stdout[:80]={stdout_preview[:80]!r}"
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Command classification heuristics
# ---------------------------------------------------------------------------

def _is_install(command: str) -> bool:
    """True if the command is a package installation."""
    if not command:
        return False
    cmd = command.lower().lstrip()
    return any(cmd.startswith(p) for p in INSTALL_PREFIXES)


def _is_server(command: str) -> bool:
    """
    True if the command launches a long-running web server.

    Install commands take priority — `pip install uvicorn` is NOT a server
    launch even though it mentions uvicorn. We check that here so callers
    that use _is_server() in isolation also get correct behavior.
    """
    if not command:
        return False
    if _is_install(command):
        return False
    return any(p.search(command) for p in SERVER_PATTERNS)


# ---------------------------------------------------------------------------
# Server launch + HTTP probe
# ---------------------------------------------------------------------------

def _extract_port(command: str) -> int:
    """
    Best-effort port extraction from a server-launch command.

    Handles:
      --port 8080 / --port=8080
      -p 8080
      http.server 8888  (Python's stdlib HTTP server takes port as bare arg)
      :8080  (e.g. host:port)
    Falls back to SERVER_DEFAULT_PORT.
    """
    if not command:
        return SERVER_DEFAULT_PORT

    # --port 8080 / --port=8080
    m = re.search(r"--port[=\s]+(\d{2,5})", command)
    if m:
        return int(m.group(1))

    # -p 8080  (must be preceded by whitespace or start of string)
    m = re.search(r"(?:^|\s)-p\s+(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    # http.server 8888  (Python's stdlib HTTP server)
    # MUST come before the generic :port check, otherwise we'd miss it
    m = re.search(r"\bhttp\.server\s+(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    # host:port (e.g. 0.0.0.0:8000)
    m = re.search(r":(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    return SERVER_DEFAULT_PORT


def _probe_http_endpoint(
    port: int,
    paths: Tuple[str, ...] = SERVER_PROBE_PATHS,
) -> Tuple[bool, str, int, str]:
    """
    Try each candidate path; return (success, path_used, status_code, body_excerpt).

    Strategy:
      - Prefer 2xx/3xx responses (real "app is working" signal)
      - Accept 4xx as routing-proof fallback (server is up, just no route at path)
      - Reject 5xx and connection errors
    """
    routing_fallback: Optional[Tuple[bool, str, int, str]] = None

    for path in paths:
        url = f"http://127.0.0.1:{port}{path}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=SERVER_PROBE_TIMEOUT_SEC) as resp:
                body = resp.read(2048).decode("utf-8", errors="replace")
                if 200 <= resp.status < 400:
                    return True, path, resp.status, body[:300]
                # 2xx fallback (non-200 OK) — record but keep looking for a 200
                routing_fallback = (True, path, resp.status, body[:300])
        except urllib.error.HTTPError as exc:
            # 4xx still proves the server is routing
            if 400 <= exc.code < 500 and routing_fallback is None:
                routing_fallback = (
                    True, path, exc.code, str(exc.reason)[:300]
                )
            # 5xx → don't treat as success
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            # Connection refused / timed out → server not ready yet, keep trying
            continue
        except Exception as exc:
            # Defensive: any other error → log and skip
            logger.debug("Probe %s raised %s; skipping", url, exc)
            continue

    return routing_fallback or (False, "", 0, "")


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """
    Kill the launched server process and ALL its children, cross-platform.

    Windows: taskkill /F /T → forcefully terminate process tree
    POSIX:   os.killpg with SIGKILL → kill process group
    """
    if proc.poll() is not None:
        return  # Already exited

    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as exc:
        # Process already gone, or we don't have permission — log and move on
        logger.debug("Server termination skipped: %s", exc)
    except Exception as exc:
        # Last-resort defensive
        logger.warning("Unexpected error terminating process tree: %s", exc)


def _run_server_and_probe(
    command: str,
    log,
) -> Tuple[bool, str, str, str]:
    """
    Background-launch a server, probe its HTTP port, terminate it.

    Returns:
        (success, stdout, stderr, summary)

    NEVER RAISES. All exceptions are caught and translated to a failed result.
    """
    try:
        workspace = ensure_workspace()
        cmd = normalize_command(command.strip())
        port = _extract_port(cmd)
    except Exception as exc:
        return False, "", f"Setup failed: {exc}", "setup_failed"

    log(f"[server] starting on port {port}...")

    popen_kwargs: Dict[str, Any] = dict(
        shell=True,
        cwd=str(workspace),
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

    # ─── Launch ──────────────────────────────────────────────────────────
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except (OSError, ValueError) as exc:
        return False, "", f"Failed to launch server: {exc}", "launch_failed"
    except Exception as exc:
        logger.exception("Unexpected Popen failure")
        return False, "", f"Server launch crashed: {exc}", "launch_crashed"

    # ─── Poll for "is it listening yet?" ─────────────────────────────────
    deadline = time.monotonic() + SERVER_STARTUP_WAIT_SEC
    probe_ok = False
    probe_path = ""
    status_code = 0
    body = ""

    try:
        while time.monotonic() < deadline:
            # Did the process die before becoming ready?
            if proc.poll() is not None:
                log("[server] process exited before becoming ready")
                break

            probe_ok, probe_path, status_code, body = _probe_http_endpoint(port)
            if probe_ok:
                break

            time.sleep(SERVER_PROBE_INTERVAL_SEC)

        # ─── Build summary based on probe outcome ────────────────────────
        if probe_ok:
            summary = (
                f"GET {probe_path} → HTTP {status_code} "
                f"(body[:80]={body[:80]!r})"
            )
            log(f"[server] ✓ endpoint responded: {summary}")
            return True, body, "", summary

        # Probe failed — gather what we can from the process
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        except Exception as exc:
            logger.debug("communicate() failed: %s", exc)
            out, err = "", ""

        if proc.returncode is not None and proc.returncode != 0:
            reason = "exited_with_error"
        elif proc.returncode is not None:
            reason = "exited_unexpectedly"
        else:
            reason = "no_response_within_timeout"

        return (
            False,
            (out or "")[:1500],
            (err or "")[:1500] or f"No HTTP response on port {port}",
            reason,
        )

    finally:
        # ALWAYS terminate the server, even on probe success
        # (we've already verified it works; no need to keep it running)
        try:
            _terminate_process_tree(proc)
        except Exception as exc:
            # Termination failure is non-fatal — the OS will clean up eventually
            logger.warning("Failed to terminate server cleanly: %s", exc)


# ---------------------------------------------------------------------------
# Defensive accessor for code-generator output
# ---------------------------------------------------------------------------

def _safe_get_str(d: dict, key: str, default: str = "") -> str:
    """Get a key from a dict, coerce to stripped str, fall back to default."""
    value = d.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


# ---------------------------------------------------------------------------
# Self-test (structural, no LLM / no real server launches)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Web Executor (structural tests, no LLM / no real servers)...\n")

    # ───────────────────────────────────────────────────────────────
    # Test 1: _is_install recognizes install commands
    # ───────────────────────────────────────────────────────────────
    assert _is_install("pip install -r requirements.txt")
    assert _is_install("pip install fastapi uvicorn")
    assert _is_install("pip3 install httpx")
    assert _is_install("python -m pip install poetry")
    assert _is_install("npm install")
    assert _is_install("npm install react")
    assert _is_install("yarn install")
    assert _is_install("pnpm i ")
    # Negative cases
    assert not _is_install("python main.py")
    assert not _is_install("uvicorn main:app")
    assert not _is_install("pip --version")
    assert not _is_install("")
    print("✓ _is_install recognizes pip/npm/yarn/pnpm install patterns")

    # ───────────────────────────────────────────────────────────────
    # Test 2: _is_server recognizes server-launch commands
    # ───────────────────────────────────────────────────────────────
    assert _is_server("uvicorn main:app")
    assert _is_server("python -m uvicorn main:app --port 8000")
    assert _is_server("gunicorn app:app -w 4")
    assert _is_server("hypercorn main:app")
    assert _is_server("flask run --port 5000")
    assert _is_server("streamlit run app.py")
    assert _is_server("python -m http.server 8080")
    # Negative cases — must NOT confuse install with server
    assert not _is_server("python main.py")
    assert not _is_server("pip install uvicorn")    # install priority
    assert not _is_server("pip3 install gunicorn")  # install priority
    assert not _is_server("./uvicorn-config-generator")   # similar word
    assert not _is_server("")
    print("✓ _is_server recognizes servers AND respects install priority")

    # ───────────────────────────────────────────────────────────────
    # Test 3: _extract_port handles all common formats
    # ───────────────────────────────────────────────────────────────
    assert _extract_port("uvicorn main:app --port 8080") == 8080
    assert _extract_port("uvicorn main:app --port=9000") == 9000
    assert _extract_port("flask run -p 5000") == 5000
    assert _extract_port("python -m http.server 8888") == 8888    # bare arg
    assert _extract_port("python -m http.server") == SERVER_DEFAULT_PORT
    assert _extract_port("uvicorn main:app --host 0.0.0.0:8001") == 8001
    # No port → default
    assert _extract_port("uvicorn main:app") == SERVER_DEFAULT_PORT
    assert _extract_port("") == SERVER_DEFAULT_PORT
    print(
        f"✓ _extract_port handles --port, -p, http.server N, host:port "
        f"(default={SERVER_DEFAULT_PORT})"
    )

    # ───────────────────────────────────────────────────────────────
    # Test 4: _safe_get_str handles LLM output oddities
    # ───────────────────────────────────────────────────────────────
    assert _safe_get_str({"k": "value"}, "k") == "value"
    assert _safe_get_str({"k": None}, "k") == ""
    assert _safe_get_str({"k": 8000}, "k") == "8000"
    assert _safe_get_str({}, "missing", default="fallback") == "fallback"
    print("✓ _safe_get_str handles None, int, missing keys")

    # ───────────────────────────────────────────────────────────────
    # Test 5: VALID_OPERATIONS set is correct
    # ───────────────────────────────────────────────────────────────
    assert OP_CREATE_FILE in VALID_OPERATIONS
    assert OP_RUN_COMMAND in VALID_OPERATIONS
    assert OP_VERIFY in VALID_OPERATIONS
    assert "exfiltrate_data" not in VALID_OPERATIONS
    print("✓ VALID_OPERATIONS contains expected operations")

    # ───────────────────────────────────────────────────────────────
    # Test 6: web_executor_node handles empty plan
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": [],
        "current_step": 0,
        "logs": [],
        "files": [],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = web_executor_node(state)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    assert delta.get("final_status") == STATUS_FAILED
    print("✓ empty plan → terminal failure delta")

    # ───────────────────────────────────────────────────────────────
    # Test 7: web_executor_node handles all-steps-done
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": ["step1", "step2"],
        "current_step": 2,
        "logs": [],
        "files": ["main.py", "requirements.txt"],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = web_executor_node(state)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    print("✓ all-steps-done → terminal delta")

    # ───────────────────────────────────────────────────────────────
    # Test 8: web_executor_node handles None / non-dict state
    # ───────────────────────────────────────────────────────────────
    delta = web_executor_node(None)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True

    delta = web_executor_node("not a state")
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    print("✓ None / non-dict state → graceful failure")

    # ───────────────────────────────────────────────────────────────
    # Test 9: Timeout configuration is sane
    # ───────────────────────────────────────────────────────────────
    assert INSTALL_TIMEOUT > DEFAULT_CMD_TIMEOUT, \
        "Install timeout should exceed default"
    assert SERVER_STARTUP_WAIT_SEC > 0
    assert SERVER_PROBE_TIMEOUT_SEC > 0
    assert SERVER_PROBE_INTERVAL_SEC > 0
    print(
        f"✓ Timeout config: install={INSTALL_TIMEOUT}s, "
        f"regular={DEFAULT_CMD_TIMEOUT}s, server-wait={SERVER_STARTUP_WAIT_SEC}s"
    )

    # ───────────────────────────────────────────────────────────────
    # Test 10: _probe_http_endpoint returns sensible defaults when no server
    # ───────────────────────────────────────────────────────────────
    # No real server running → should fail gracefully, no exceptions
    ok, path, status, body = _probe_http_endpoint(
        port=59999,    # unlikely to be in use
        paths=("/test",),
    )
    assert ok is False
    assert path == ""
    assert status == 0
    print("✓ _probe_http_endpoint returns clean failure when server is absent")

    # ───────────────────────────────────────────────────────────────
    # Test 11: StepResult contract — _run_web_step never raises
    # ───────────────────────────────────────────────────────────────
    new_logs = []
    log = make_logger(new_logs)
    try:
        result = _run_web_step(
            step_description="",
            requirement="test",
            plan=[""],
            files_so_far=[],
            log=log,
        )
        assert isinstance(result, StepResult), \
            f"_run_web_step returned {type(result).__name__} not StepResult"
        print("✓ _run_web_step returns StepResult (even on edge cases)")
    except Exception as exc:
        raise AssertionError(
            f"_run_web_step raised instead of returning StepResult: {exc}"
        )

    print("\n✓ All 11 structural self-tests passed.")
    print("\nNote: Integration testing (real server launch + HTTP probe) happens")
    print("when you run `python main.py` with a FastAPI / Flask prompt.")