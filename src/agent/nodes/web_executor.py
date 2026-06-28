"""
Web Executor

Handles web applications (FastAPI, Flask, Streamlit, Django, http.server)
with three command modes:
- install: long timeout for pip/npm dependency installation
- server:  background launch + HTTP endpoint probing for verification
- regular: standard timeout for tests, scripts, etc.

The HTTP probe is the differentiator from simple/compiled executors — we
don't just check that the process started, we hit the *planned* endpoint
and confirm it responds correctly. That's real verification.

Day 3 update:
- The probe now extracts the target endpoint from the planned curl/test
  command (e.g. `curl http://127.0.0.1:8000/multiply?a=5&b=10`) and probes
  THAT endpoint, not a hardcoded /docs. This means broken endpoints
  actually surface as failures and trigger the Error Analyzer → Fix
  Generator recovery loop.
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
from typing import List, Optional, Tuple

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
# Configuration
# ---------------------------------------------------------------------------

INSTALL_TIMEOUT_SEC = 180        # pip / npm install can take 2+ minutes
DEFAULT_TIMEOUT_SEC = 30
MAX_OUTPUT_PREVIEW = 200

# Server probe configuration
SERVER_DEFAULT_PORT = 8000
SERVER_STARTUP_WAIT_SEC = 10
SERVER_PROBE_TIMEOUT_SEC = 3.0
SERVER_PROBE_INTERVAL_SEC = 0.5

# Fallback probe paths used ONLY when no planned endpoint can be parsed
# from the command. When the plan contains an explicit curl URL we probe
# THAT path instead (see _extract_probe_target).
SERVER_FALLBACK_PROBE_PATHS = (
    "/docs",
    "/health",
    "/",
    "/ping",
)

VALID_OPERATIONS = {"create_file", "update_file", "execute_command", "verify"}

# Prefixes that identify package install commands
INSTALL_PREFIXES = (
    "pip install", "pip3 install", "python -m pip install",
    "npm install", "npm i ", "yarn ", "yarn install",
    "pnpm install", "pnpm i ", "uv pip install",
)

# Patterns that identify server-launch commands. Each requires the server
# name to be followed by whitespace or end-of-line so `uvicorn-config-gen`
# doesn't accidentally match.
SERVER_PATTERNS = (
    re.compile(r"\buvicorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bgunicorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bhypercorn(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bdaphne(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bstreamlit\s+run\b", re.IGNORECASE),
    re.compile(r"\bpython\s+-m\s+http\.server\b", re.IGNORECASE),
)

# Regex to pull a planned target URL out of a command (curl, wget, etc.)
_PLANNED_URL_RE = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1)(?::(\d+))?(/[^\s'\"`)]*)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def web_executor_node(state: AgentState) -> dict:
    """Execute one step for a web-application project."""
    new_logs: List[str] = []
    log = make_logger(new_logs)
    log("--- WEB EXECUTOR ---")

    s = extract_state_basics(state)

    # Terminal-condition guards
    delta = check_empty_plan(s, new_logs, log)
    if delta is not None:
        return delta

    delta = check_all_steps_done(s, new_logs, log)
    if delta is not None:
        return delta

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
# Step runner — three command modes
# ---------------------------------------------------------------------------

def _run_web_step(
    *,
    step_description: str,
    requirement: str,
    plan: List[str],
    files_so_far: List[str],
    log,
) -> StepResult:
    """Generate the step's action via code_generator, then dispatch by mode."""
    result = StepResult()

    log("[plan] generating execution plan...")
    plan_dict = generate_plan_for_step(
        step_description, requirement, plan, files_so_far,
    )

    operation = (plan_dict.get("operation") or "").strip()
    file_path = (plan_dict.get("file_path") or "").strip()
    file_content = plan_dict.get("file_content") or ""
    command = (plan_dict.get("command") or "").strip()

    result.plan = plan_dict
    result.command = command
    result.file_path = file_path
    result.file_content = file_content

    log(
        f"[plan] operation='{operation}', file_path='{file_path or 'none'}', "
        f"has_command={bool(command)}"
    )

    # verify is a logical no-op
    if operation == "verify":
        log("[verify] no-op step (logical checkpoint)")
        result.success = True
        result.verification_summary = "verify step (no action required)"
        return result

    if operation and operation not in VALID_OPERATIONS:
        msg = f"Unknown operation {operation!r}; expected one of {sorted(VALID_OPERATIONS)}"
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    if not write_file_if_needed(file_path, file_content, result, log):
        return result

    if not command:
        log("[run] skipped (no command)")
        result.success = True
        return result

    # Dispatch to the right mode. Install check first so `pip install uvicorn`
    # isn't misclassified as a server launch.
    if _is_install(command):
        return _handle_install(command, result, log)
    if _is_server(command):
        return _handle_server(command, result, log)
    return _handle_regular(command, result, log)


# ---------------------------------------------------------------------------
# Command mode handlers
# ---------------------------------------------------------------------------

def _handle_install(command: str, result: StepResult, log) -> StepResult:
    """Run an install command with the long install timeout."""
    log(f"[install] {command}  (timeout={INSTALL_TIMEOUT_SEC}s)")
    ok, stdout, stderr = execute_command(command, timeout=INSTALL_TIMEOUT_SEC)
    result.stdout, result.stderr = stdout or "", stderr or ""

    if not ok:
        preview = (stderr or stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
        log(f"[install] ✗ failed: {preview!r}", level=logging.ERROR)
        result.error_message = f"Install failed: {preview}"
        return result

    preview = (stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
    log(f"[install] ✓ done. stdout[:120]={preview[:120]!r}")
    result.verification_summary = f"installed OK; stdout[:80]={preview[:80]!r}"
    result.success = True
    return result


def _handle_server(command: str, result: StepResult, log) -> StepResult:
    """Launch in background, probe HTTP endpoint, terminate."""
    log(f"[server] launching and probing — {command}")
    ok, stdout, stderr, summary = _run_server_and_probe(command, log)
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
    result.error_message = f"Server verification failed ({summary}): {stderr.strip()[:200]}"
    return result


def _handle_regular(command: str, result: StepResult, log) -> StepResult:
    """Run a non-install, non-server command with standard timeout."""
    log(f"[run] {command}  (timeout={DEFAULT_TIMEOUT_SEC}s)")
    ok, stdout, stderr = execute_command(command, timeout=DEFAULT_TIMEOUT_SEC)
    result.stdout, result.stderr = stdout or "", stderr or ""

    if not ok:
        preview = (stderr or stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
        log(f"[run] ✗ failed: {preview!r}", level=logging.ERROR)
        result.error_message = f"Command failed: {preview}"
        return result

    preview = (stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
    log(f"[run] ✓ exit=0  stdout[:120]={preview[:120]!r}")
    result.verification_summary = f"exit=0; stdout[:80]={preview[:80]!r}"
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------

def _is_install(command: str) -> bool:
    """True if the command is a package install."""
    if not command:
        return False
    cmd = command.lower().lstrip()
    return any(cmd.startswith(p) for p in INSTALL_PREFIXES)


def _is_server(command: str) -> bool:
    """
    True if the command launches a long-running web server.

    Install commands take priority: `pip install uvicorn` is NOT a server
    launch even though it mentions uvicorn.
    """
    if not command or _is_install(command):
        return False
    return any(p.search(command) for p in SERVER_PATTERNS)


# ---------------------------------------------------------------------------
# Server launch + HTTP probe
# ---------------------------------------------------------------------------

def _extract_port(command: str) -> int:
    """Pull a port number out of a server-launch command, with fallback to default."""
    if not command:
        return SERVER_DEFAULT_PORT

    # --port 8080 or --port=8080
    m = re.search(r"--port[=\s]+(\d{2,5})", command)
    if m:
        return int(m.group(1))

    # -p 8080 (preceded by whitespace or start)
    m = re.search(r"(?:^|\s)-p\s+(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    # http.server 8888 — must come before generic :port to win
    m = re.search(r"\bhttp\.server\s+(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    # host:port like 0.0.0.0:8000
    m = re.search(r":(\d{2,5})\b", command)
    if m:
        return int(m.group(1))

    return SERVER_DEFAULT_PORT


def _extract_probe_target(command: str, default_port: int) -> Tuple[Tuple[str, ...], int, bool]:
    """
    Parse the planned command for a target URL (e.g. `curl http://127.0.0.1:8000/multiply?a=5&b=10`)
    and return the probe paths to try plus the port.

    Returns:
        (paths_to_try, port, planned)
        - paths_to_try: ordered tuple of paths to probe
        - port: port discovered (from URL if present, else default_port)
        - planned: True if we found an explicit endpoint in the command,
                   False if we're falling back to generic paths

    When `planned=True`, the executor must treat 4xx/5xx as FAILURE because
    we know exactly what the user wanted hit. When `planned=False`, we keep
    the lenient routing-fallback behavior (a 4xx on /docs still means the
    server is up).
    """
    if not command:
        return SERVER_FALLBACK_PROBE_PATHS, default_port, False

    match = _PLANNED_URL_RE.search(command)
    if not match:
        return SERVER_FALLBACK_PROBE_PATHS, default_port, False

    port = int(match.group(1)) if match.group(1) else default_port
    path = match.group(2) or "/"
    return (path,), port, True


def _probe_http_endpoint(
    port: int,
    paths: Tuple[str, ...],
    strict: bool,
) -> Tuple[bool, str, int, str]:
    """
    Try each probe path; return (success, path_used, status_code, body_excerpt).

    Strategy when strict=False (no planned endpoint):
    - 2xx/3xx = real success
    - 4xx = accept as routing proof (server is up, just no route at that path)
    - 5xx and network errors = failure

    Strategy when strict=True (planned endpoint from the command):
    - 2xx/3xx = success
    - 4xx, 5xx, network errors = failure (we know what should have worked)
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
                # 4xx/5xx on a 200-class response (rare)
                if not strict and routing_fallback is None:
                    routing_fallback = (True, path, resp.status, body[:300])
        except urllib.error.HTTPError as exc:
            # Server is up but returned an error code
            if strict:
                # Planned endpoint MUST return success; surface the error
                try:
                    body = exc.read(2048).decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc.reason)
                # 5xx = clear failure; 4xx on a planned URL = also failure
                return False, path, exc.code, body[:300]
            if 400 <= exc.code < 500 and routing_fallback is None:
                routing_fallback = (True, path, exc.code, str(exc.reason)[:300])
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            # Server not ready yet — keep trying
            continue

    return routing_fallback or (False, "", 0, "")


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Kill the launched server and all its children, cross-platform."""
    if proc.poll() is not None:
        return  # Already exited

    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as exc:
        logger.debug("Server termination skipped: %s", exc)


def _run_server_and_probe(
    command: str,
    log,
) -> Tuple[bool, str, str, str]:
    """
    Background-launch a server, probe its HTTP endpoint, terminate.

    The probe target is extracted from the planned command when possible
    (e.g. `... && curl http://127.0.0.1:8000/multiply?a=5&b=10`), so we
    actually validate the endpoint the user asked for — not just /docs.

    Returns (success, stdout, stderr, summary).
    """
    workspace = ensure_workspace()
    cmd = normalize_command(command.strip())

    # Port comes from the server-launch flags; probe target comes from the
    # planned curl/test URL inside the same command.
    port = _extract_port(cmd)
    probe_paths, probe_port, planned = _extract_probe_target(cmd, default_port=port)

    if planned:
        log(f"[server] starting on port {probe_port}... "
            f"(planned endpoint: {probe_paths[0]})")
    else:
        log(f"[server] starting on port {probe_port}... "
            f"(no planned endpoint; will probe {list(probe_paths)})")

    popen_kwargs = dict(
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

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except (OSError, ValueError) as exc:
        return False, "", f"Failed to launch: {exc}", "launch_failed"

    # Poll for readiness
    deadline = time.monotonic() + SERVER_STARTUP_WAIT_SEC
    probe_ok = False
    probe_path = ""
    status_code = 0
    body = ""

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log("[server] process exited before becoming ready")
                break
            probe_ok, probe_path, status_code, body = _probe_http_endpoint(
                probe_port, probe_paths, strict=planned,
            )
            # If planned and we got a definitive non-2xx, stop early —
            # the endpoint is reachable but broken; no point waiting.
            if probe_ok:
                break
            if planned and status_code and status_code >= 400:
                break
            time.sleep(SERVER_PROBE_INTERVAL_SEC)

        if probe_ok:
            summary = (
                f"GET {probe_path} → HTTP {status_code} "
                f"(body[:80]={body[:80]!r})"
            )
            log(f"[server] ✓ endpoint responded: {summary}")
            return True, body, "", summary

        # Probe failed — gather diagnostics from the process
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out, err = "", ""

        # Decide failure reason
        if planned and status_code and status_code >= 400:
            reason = (
                f"GET {probe_path} → HTTP {status_code} "
                f"(body[:80]={body[:80]!r})"
            )
        elif proc.returncode is not None and proc.returncode != 0:
            reason = "exited_with_error"
        elif proc.returncode is not None:
            reason = "exited_unexpectedly"
        else:
            reason = "no_response_within_timeout"

        # When the endpoint returned a definitive 4xx/5xx, surface the
        # response body in stderr so the Error Analyzer can read it.
        stderr_payload = (err or "")[:1500]
        if planned and status_code and status_code >= 400 and not stderr_payload:
            stderr_payload = (
                f"Endpoint {probe_path} returned HTTP {status_code}. "
                f"Body: {body[:400]}"
            )
        if not stderr_payload:
            stderr_payload = f"No HTTP response on port {probe_port}"

        return (
            False,
            (out or "")[:1500],
            stderr_payload,
            reason,
        )

    finally:
        # Always kill the server, even on success. We've verified it works;
        # leaving it running would block the next prompt.
        _terminate_process_tree(proc)