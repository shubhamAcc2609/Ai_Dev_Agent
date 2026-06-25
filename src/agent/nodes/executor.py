"""
Executor Node

Orchestrates the execution of a single step in the plan by:
1. Using Code Generator (with full task context) to create an execution plan
2. Using File Manager to create/update files
3. Using Execution Manager to run commands (with platform-aware normalization)
4. Using Error Analyzer to classify failures
5. Using Fix Generator to auto-fix and retry recoverable errors
6. Returning structured state updates for the agent graph

Special handling:
- Server-launch commands (uvicorn, gunicorn, flask run, etc.) are launched
  in the background, probed via HTTP, and then terminated cleanly. Success
  requires an actual endpoint response, not just a startup log line.
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent.state import AgentState
from src.tools.code_generator import generate_execution_plan
from src.tools.error_analyzer import analyze_execution_failure
from src.tools.execution_manager import (
    IS_WINDOWS,
    ensure_workspace,
    execute_command,
    normalize_command,
)
from src.tools.file_manager import create_or_update_file
from src.tools.fix_generator import apply_fixes, generate_fixes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
MAX_FIX_ATTEMPTS_PER_STEP = 2
DEFAULT_COMMAND_TIMEOUT = 30
LONG_COMMAND_TIMEOUT = 180

LONG_RUNNING_PREFIXES = (
    "pip install", "pip3 install",
    "python -m pip install",
    "npm install", "npm i", "yarn",
    "pytest", "go build", "cargo build",
    "docker build",
)

# Patterns that identify long-running server processes
SERVER_LAUNCH_PATTERNS = (
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bhypercorn\b", re.IGNORECASE),
    re.compile(r"\bdaphne\b", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bstreamlit\s+run\b", re.IGNORECASE),
    re.compile(r"\bpython\s+-m\s+http\.server\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+(?:run\s+)?(?:start|dev|serve)\b", re.IGNORECASE),
    re.compile(r"\byarn\s+(?:start|dev|serve)\b", re.IGNORECASE),
)

# Server probe configuration
SERVER_DEFAULT_PORT = 8000
SERVER_STARTUP_WAIT_SEC = 10        # max time to wait for server to listen
SERVER_PROBE_TIMEOUT_SEC = 3.0      # per-request timeout
SERVER_PROBE_PATHS = (
    "/", "/health", "/healthz", "/ping",
    "/docs", "/openapi.json",
)

# Status values
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"


# ---------------------------------------------------------------------------
# Internal step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    success: bool = False
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    file_path: str = ""
    file_content: str = ""
    files_created: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    verification_summary: str = ""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _delta(**updates) -> dict:
    """Build a state-update dict, dropping ``...`` sentinels."""
    return {k: v for k, v in updates.items() if v is not ...}


def _pick_timeout(command: str) -> int:
    lowered = command.lower().lstrip()
    for prefix in LONG_RUNNING_PREFIXES:
        if lowered.startswith(prefix):
            return LONG_COMMAND_TIMEOUT
    return DEFAULT_COMMAND_TIMEOUT


def _is_server_launch(command: str) -> bool:
    """True if the command starts a long-running server."""
    return any(p.search(command) for p in SERVER_LAUNCH_PATTERNS)


# ---------------------------------------------------------------------------
# Server probing
# ---------------------------------------------------------------------------

def _extract_port(command: str) -> int:
    """Best-effort port extraction from a server-launch command."""
    m = re.search(r"--port[=\s]+(\d{2,5})", command)
    if m:
        return int(m.group(1))
    m = re.search(r"-p\s+(\d{2,5})", command)
    if m:
        return int(m.group(1))
    m = re.search(r":(\d{2,5})\b", command)
    if m:
        return int(m.group(1))
    return SERVER_DEFAULT_PORT


def _probe_http_endpoint(
    port: int,
    paths: Tuple[str, ...] = SERVER_PROBE_PATHS,
) -> Tuple[bool, str, int, str]:
    """
    Try candidate paths; return (success, path_used, status_code, body_snippet).

    A 4xx response is still proof the server is up and routing requests.
    """
    for path in paths:
        url = f"http://127.0.0.1:{port}{path}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=SERVER_PROBE_TIMEOUT_SEC) as resp:
                body = resp.read(2048).decode("utf-8", errors="replace")
                return True, path, resp.status, body[:300]
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                return True, path, exc.code, str(exc.reason)[:300]
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            continue
    return False, "", 0, ""


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Kill the launched server and its children cross-platform."""
    if proc.poll() is not None:
        return
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
        logger.debug("Server termination skipped: %s", exc)


def _run_server_and_probe(command: str, log) -> Tuple[bool, str, str, str]:
    """
    Start a server in the background, probe its HTTP port, then stop it.

    Returns:
        (success, stdout_excerpt, stderr_excerpt, summary)
    """
    workspace = ensure_workspace()
    cmd = normalize_command(command.strip())
    port = _extract_port(cmd)

    log(f"    Starting server in background (port={port})...")

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

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except OSError as exc:
        return False, "", f"Failed to launch: {exc}", "launch_failed"

    deadline = time.monotonic() + SERVER_STARTUP_WAIT_SEC
    probe_ok = False
    probe_path = ""
    status_code = 0
    body = ""

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log("    Server process exited prematurely")
                break
            probe_ok, probe_path, status_code, body = _probe_http_endpoint(port)
            if probe_ok:
                break
            time.sleep(0.5)

        if probe_ok:
            summary = (
                f"GET {probe_path} → HTTP {status_code} "
                f"(body[:80]={body[:80]!r})"
            )
            log(f"    ✓ Endpoint responded: {summary}")
            return True, body, "", summary

        # Probe failed — collect what's available from the process
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
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
        _terminate_process_tree(proc)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def executor_node(state: AgentState) -> dict:
    """
    Execute the current step in the plan and return state updates.

    The returned dict is *only* the delta to merge into AgentState — we never
    mutate the caller's lists in place.
    """
    plan: List[str] = list(state.get("plan", []) or [])
    current_step: int = int(state.get("current_step", 0) or 0)
    existing_logs: List[str] = list(state.get("logs", []) or [])
    existing_files: List[str] = list(state.get("files", []) or [])
    retry_count: int = int(state.get("retry_count", 0) or 0)
    prior_last_error: Optional[str] = state.get("last_error")
    requirement: str = (state.get("requirement") or "").strip()

    new_logs: List[str] = []

    def log(msg: str, level: int = logging.INFO) -> None:
        logger.log(level, msg)
        new_logs.append(msg)

    log("--- EXECUTOR NODE ---")

    # ----- Guard: empty plan ---------------------------------------------
    if not plan:
        log("No plan available; nothing to execute.")
        return _delta(
            logs=existing_logs + new_logs,
            files=existing_files,
            is_complete=True,
            current_step=0,
            retry_count=0,
            last_error="No plan provided",
            final_status=STATUS_FAILED,
        )

    # ----- All steps done -------------------------------------------------
    if current_step < 0 or current_step >= len(plan):
        had_errors = bool(prior_last_error)
        status = STATUS_FAILED if had_errors else STATUS_SUCCESS
        log(f"All {len(plan)} steps processed. Final status: {status}")
        return _delta(
            logs=existing_logs + new_logs,
            files=_dedupe_preserve_order(existing_files),
            is_complete=True,
            current_step=current_step,
            retry_count=0,
            last_error=prior_last_error,
            plan_feedback=None,
            final_status=status,
        )

    step_description = plan[current_step]
    log(f"Executing step {current_step + 1}/{len(plan)}: {step_description}")

    # ----- Run the step ---------------------------------------------------
    result: Optional[StepResult] = None
    try:
        result = _execute_step(
            step_description=step_description,
            log=log,
            requirement=requirement,
            plan_overview=plan,
            files_so_far=existing_files,
        )
    except Exception as exc:
        logger.exception("Catastrophic error in executor")
        log(f"CRITICAL ERROR: {exc}", level=logging.ERROR)
        return _handle_step_failure(
            base_logs=existing_logs + new_logs,
            base_files=existing_files,
            retry_count=retry_count,
            current_step=current_step,
            total_steps=len(plan),
            step_description=step_description,
            error_msg=str(exc),
            stdout="",
            stderr=str(exc),
            command="<executor crashed>",
            file_path="",
            file_content="",
        )

    # ----- Defensive guard ------------------------------------------------
    if result is None:
        msg = "Internal error: _execute_step returned None"
        logger.error(msg)
        log(f"CRITICAL ERROR: {msg}", level=logging.ERROR)
        return _handle_step_failure(
            base_logs=existing_logs + new_logs,
            base_files=existing_files,
            retry_count=retry_count,
            current_step=current_step,
            total_steps=len(plan),
            step_description=step_description,
            error_msg=msg,
            stdout="",
            stderr=msg,
            command="<executor returned None>",
            file_path="",
            file_content="",
        )

    # ----- Branch on result ----------------------------------------------
    if result.success:
        merged_files = _dedupe_preserve_order(
            existing_files + result.files_created
        )
        new_step = current_step + 1
        is_done = new_step >= len(plan)

        verification_note = (
            f" [verified: {result.verification_summary}]"
            if result.verification_summary
            else ""
        )
        log(
            f"✓ Step {current_step + 1} completed.{verification_note} "
            f"Files this step: {result.files_created or 'none'}"
        )

        if is_done:
            final_status = (
                STATUS_SUCCESS if not prior_last_error else STATUS_FAILED
            )
            log(f"All steps complete. Final status: {final_status}")
            return _delta(
                logs=existing_logs + new_logs,
                files=merged_files,
                current_step=new_step,
                retry_count=0,
                last_error=prior_last_error,
                plan_feedback=None,
                is_complete=True,
                final_status=final_status,
            )

        return _delta(
            logs=existing_logs + new_logs,
            files=merged_files,
            current_step=new_step,
            retry_count=0,
            last_error=None,
            plan_feedback=None,
            is_complete=False,
        )

    return _handle_step_failure(
        base_logs=existing_logs + new_logs,
        base_files=_dedupe_preserve_order(existing_files + result.files_created),
        retry_count=retry_count,
        current_step=current_step,
        total_steps=len(plan),
        step_description=step_description,
        error_msg=result.error_message,
        stdout=result.stdout,
        stderr=result.stderr,
        command=result.command,
        file_path=result.file_path,
        file_content=result.file_content,
    )


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _execute_step(
    *,
    step_description: str,
    log,
    requirement: str = "",
    plan_overview: Optional[List[str]] = None,
    files_so_far: Optional[List[str]] = None,
) -> StepResult:
    """
    Translate a step description into concrete actions.

    GUARANTEE: This function always returns a StepResult. Every code path
    ends with `return result`.
    """
    log("[1/3] Code Generator: generating execution plan...")
    plan = generate_execution_plan(
        step_description=step_description,
        requirement=requirement,
        plan_overview=plan_overview,
        files_so_far=files_so_far,
    )

    operation = (plan.get("operation") or "").strip()
    file_path = (plan.get("file_path") or "").strip()
    file_content = plan.get("file_content") or ""
    command = (plan.get("command") or "").strip()
    log(
        f"    operation={operation!r} file_path={file_path!r} "
        f"has_command={bool(command)}"
    )

    result = StepResult(
        plan=plan,
        command=command,
        file_path=file_path,
        file_content=file_content,
    )

    # --- verify is a logical no-op for execution -------------------------
    if operation == "verify":
        log("[2/3] verify operation: no file or command action required")
        result.success = True
        result.verification_summary = "manual verification (no-op step)"
        return result

    # --- File write -------------------------------------------------------
    if file_path:
        log(f"[2/3] File Manager: writing {file_path}")
        ok, err, created = create_or_update_file(file_path, file_content)
        if not ok:
            log(f"    ✗ File write failed: {err}", level=logging.ERROR)
            result.error_message = f"File creation failed: {err}"
            result.stderr = err
            return result
        result.files_created.extend(created)
        log(f"    ✓ Wrote: {created}")
    else:
        log("[2/3] File Manager: skipped (no file_path)")

    # --- Command execution ------------------------------------------------
    if command:
        if _is_server_launch(command):
            log(f"[3/3] Server-mode execution: launching and probing — {command}")
            ok, stdout, stderr, summary = _run_server_and_probe(command, log)
            result.stdout = stdout
            result.stderr = stderr
            if ok:
                log(f"    ✓ Verified live: {summary}")
                result.success = True
                result.verification_summary = summary
                return result
            log(
                f"    ✗ Server verification failed ({summary}): "
                f"stderr[:200]={stderr.strip()[:200]!r}",
                level=logging.ERROR,
            )
            result.error_message = (
                f"Server verification failed ({summary}): "
                f"{stderr.strip()[:200]}"
            )
            return result

        timeout = _pick_timeout(command)
        log(
            f"[3/3] Execution Manager: running "
            f"(timeout={timeout}s) — {command}"
        )
        ok, stdout, stderr = execute_command(command, timeout=timeout)
        result.stdout = stdout
        result.stderr = stderr

        if not ok:
            log(
                f"    ✗ Command failed: {stderr.strip()[:200]}",
                level=logging.ERROR,
            )
            result.error_message = f"Command failed: {stderr.strip()[:200]}"
            return result
        log(f"    ✓ Command ok. stdout[:120]={stdout[:120]!r}")
        result.verification_summary = f"exit=0; stdout[:80]={stdout[:80]!r}"
    else:
        log("[3/3] Execution Manager: skipped (no command)")

    result.success = True
    return result


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def _handle_step_failure(
    *,
    base_logs: List[str],
    base_files: List[str],
    retry_count: int,
    current_step: int,
    total_steps: int,
    step_description: str,
    error_msg: str,
    stdout: str,
    stderr: str,
    command: str,
    file_path: str,
    file_content: str,
) -> dict:
    """Diagnose the failure, attempt a fix when safe, and decide next action."""
    retry_count += 1
    logs = list(base_logs)
    logs.append(
        f"--- ERROR RECOVERY (attempt {retry_count}/{MAX_RETRIES}) ---"
    )
    logger.info("Error recovery attempt %d/%d", retry_count, MAX_RETRIES)

    # --- Analyze ---------------------------------------------------------
    error_analysis: Optional[Dict[str, Any]] = None
    if stdout or stderr:
        try:
            error_analysis = analyze_execution_failure(stdout, stderr, command)
            logs.append(
                f"Error Analyzer: type={error_analysis.get('error_type')} "
                f"severity={error_analysis.get('severity')} "
                f"recoverable={error_analysis.get('is_recoverable')}"
            )
            logs.append(f"  root_cause: {error_analysis.get('root_cause')}")
        except Exception as exc:
            logger.exception("Error analyzer crashed")
            logs.append(f"Error Analyzer crashed: {exc}")

    # --- Attempt a fix when safe ----------------------------------------
    fix_attempted = False
    if (
        error_analysis
        and error_analysis.get("is_recoverable")
        and retry_count <= MAX_FIX_ATTEMPTS_PER_STEP
    ):
        fix_attempted = True
        try:
            fix_plan = generate_fixes(
                error_analysis,
                code_context=file_content[:2000],
                file_path=file_path,
            )
            logs.append(
                f"Fix Generator: {len(fix_plan.get('fixes', []))} fix(es), "
                f"confidence={fix_plan.get('confidence', 'n/a')}%"
            )
            fix_ok, applied, fix_errors = apply_fixes(fix_plan)
            for desc in applied:
                logs.append(f"  applied: {desc}")
            for err in fix_errors:
                logs.append(f"  fix-error: {err}")

            if fix_ok and applied:
                logs.append(
                    f"⚠ Fixes applied; retrying step {current_step + 1} "
                    f"(retry {retry_count}/{MAX_RETRIES})"
                )
                return _delta(
                    logs=logs,
                    files=base_files,
                    current_step=current_step,
                    retry_count=retry_count,
                    last_error=error_msg,
                    plan_feedback=None,
                    is_complete=False,
                )
        except Exception as exc:
            logger.exception("Fix generator crashed")
            logs.append(f"Fix Generator crashed: {exc}")

    # --- Decide: escalate to planner or retry raw ------------------------
    if retry_count >= MAX_RETRIES:
        feedback = _build_plan_feedback(
            step_description, error_msg, error_analysis
        )
        logs.append(
            f"✗ Step {current_step + 1} failed after {MAX_RETRIES} attempts. "
            "Escalating to planner."
        )
        return _delta(
            logs=logs,
            files=base_files,
            current_step=current_step,
            retry_count=retry_count,
            last_error=error_msg,
            plan_feedback=feedback,
            is_complete=False,
        )

    if not fix_attempted:
        logs.append(
            f"⚠ Step {current_step + 1} failed "
            f"(retry {retry_count}/{MAX_RETRIES})"
        )
    return _delta(
        logs=logs,
        files=base_files,
        current_step=current_step,
        retry_count=retry_count,
        last_error=error_msg,
        plan_feedback=None,
        is_complete=False,
    )


def _build_plan_feedback(
    step_description: str,
    error_msg: str,
    error_analysis: Optional[Dict[str, Any]],
) -> str:
    """Compose a structured message for the planner to replan from."""
    parts = [f"Step '{step_description}' failed: {error_msg}"]
    if error_analysis:
        parts.append(
            f"Error type: {error_analysis.get('error_type')}; "
            f"severity: {error_analysis.get('severity')}; "
            f"root cause: {error_analysis.get('root_cause')}."
        )
        if error_analysis.get("suggested_fix"):
            parts.append(f"Suggested fix: {error_analysis['suggested_fix']}")
    parts.append("Please provide an alternative approach.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mock_state: AgentState = AgentState(
        requirement="A FastAPI endpoint that returns the current weather for a city",
        plan=[
            "Create requirements.txt listing fastapi and uvicorn",
            "Install dependencies with pip install -r requirements.txt",
            "Create main.py with a FastAPI app exposing GET /weather",
            "Run uvicorn and verify GET /docs returns HTTP 200",
        ],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False,
        last_error=None,
        retry_count=0,
        plan_feedback=None,
        user_feedback=None,
    )

    print("Testing Executor Node (Step 1)...\n")
    result = executor_node(mock_state)

    print("\n--- Executor Result ---")
    for key, value in result.items():
        if key == "logs":
            print(f"{key}:")
            for entry in value:
                print(f"  - {entry}")
        else:
            print(f"{key}: {value}")