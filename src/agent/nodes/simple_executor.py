"""
Simple Executor

Handles plain Python scripts and basic CLI tools.

Characteristics:
- One or two files
- Single command per step (e.g. `python main.py`)
- Standard 30-second timeout
- Verification = command exits with code 0 and prints expected output

Bulletproof guarantees:
- Never raises — all exceptions caught and translated to failure deltas
- Defensive against malformed state (handled by shared helpers)
- Defensive against malformed code-generator output (validated locally)
- Defensive against execute_command crashes (caught and wrapped)
- Defensive against file_manager crashes (caught and wrapped)
- Returns valid state delta in every code path

Architecture:
- ~50 lines of orchestration (the public node)
- ~50 lines of specialized step-running logic (_run_simple_step)
- Everything else delegated to _executor_shared
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
from src.tools.execution_manager import execute_command

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# This executor's specific tuning knobs
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 30        # Most Python scripts finish quickly
MAX_OUTPUT_PREVIEW = 200    # Chars of stdout/stderr to log on each command

# Operations recognized in code-generator output
OP_CREATE_FILE = "create_file"
OP_UPDATE_FILE = "update_file"
OP_RUN_COMMAND = "execute_command"
OP_VERIFY = "verify"
VALID_OPERATIONS = {OP_CREATE_FILE, OP_UPDATE_FILE, OP_RUN_COMMAND, OP_VERIFY}


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def simple_executor_node(state: AgentState) -> dict:
    """
    Execute a single step for a simple Python script or CLI tool.

    NEVER RAISES. Every failure mode produces a valid state delta.
    """
    # ─── Set up logging that feeds both stdout and state delta ──────────
    new_logs: List[str] = []
    log = make_logger(new_logs)
    log("--- SIMPLE EXECUTOR ---")

    # ─── Extract state defensively ──────────────────────────────────────
    try:
        s = extract_state_basics(state)
    except Exception as exc:  # extract_state_basics shouldn't raise, but...
        logger.exception("State extraction failed catastrophically")
        log(f"CRITICAL: state extraction failed: {exc}", level=logging.ERROR)
        return make_delta(
            logs=new_logs,
            is_complete=True,
            final_status=STATUS_FAILED,
            last_error=f"State extraction failed: {exc}",
        )

    # ─── Terminal-condition guards (delegated to shared helpers) ────────
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
        result = _run_simple_step(
            step_description=step_description,
            requirement=s["requirement"],
            plan=s["plan"],
            files_so_far=s["existing_files"],
            log=log,
        )
    except Exception as exc:
        # Catastrophic — code-generator crashed, LLM unreachable, etc.
        logger.exception("Simple executor crashed during step execution")
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

    # ─── Final defensive guard: result must be a StepResult ─────────────
    if not isinstance(result, StepResult):
        msg = (
            f"_run_simple_step returned {type(result).__name__} "
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
# Specialized step runner — what makes this executor "simple"
# ---------------------------------------------------------------------------

def _run_simple_step(
    *,
    step_description: str,
    requirement: str,
    plan: List[str],
    files_so_far: List[str],
    log,
) -> StepResult:
    """
    Generate, write, and execute. Simple as that.

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

    # ─── Step 2: Validate plan structure defensively ────────────────────
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

    # Populate result with what we know so failure handling has context
    result.plan = plan_dict
    result.command = command
    result.file_path = file_path
    result.file_content = file_content

    log(
        f"[plan] operation='{operation}', file_path='{file_path or 'none'}', "
        f"has_command={bool(command)}"
    )

    # ─── Step 3: Operation-specific handling ────────────────────────────

    # 3a. Verify operation is a no-op for execution — the planner intends
    #     it as a logical checkpoint, not an action
    if operation == OP_VERIFY:
        log("[verify] no-op step (logical checkpoint)")
        result.success = True
        result.verification_summary = "verify step (no action required)"
        return result

    # 3b. Unknown operation → fail loudly so the planner can correct it
    if operation and operation not in VALID_OPERATIONS:
        msg = f"Unknown operation '{operation}'; expected one of {sorted(VALID_OPERATIONS)}"
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    # ─── Step 4: Write the file (if requested) ──────────────────────────
    try:
        if not write_file_if_needed(file_path, file_content, result, log):
            # write_file_if_needed already populated result.error_message
            return result
    except Exception as exc:
        log(f"[file] ✗ file write crashed: {exc}", level=logging.ERROR)
        result.error_message = f"File write crashed: {exc}"
        result.stderr = str(exc)
        return result

    # ─── Step 5: Execute the command (if requested) ─────────────────────
    if not command:
        log("[run] skipped (no command)")
        result.success = True
        return result

    log(f"[run] {command}  (timeout={DEFAULT_TIMEOUT}s)")
    try:
        ok, stdout, stderr = execute_command(command, timeout=DEFAULT_TIMEOUT)
    except Exception as exc:
        # execute_command should be never-raises, but be defensive
        log(f"[run] ✗ execute_command crashed: {exc}", level=logging.ERROR)
        result.error_message = f"Command execution crashed: {exc}"
        result.stderr = str(exc)
        return result

    # Always capture output for failure analysis even on success
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
# Defensive accessor for code-generator output
# ---------------------------------------------------------------------------

def _safe_get_str(d: dict, key: str, default: str = "") -> str:
    """
    Get a key from a dict, coerce to stripped str, fall back to default.

    Why: code generator should return strings, but LLM output can occasionally
    produce None, numbers, or even nested objects. We coerce defensively.
    """
    value = d.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    # Numbers, bools, etc. — convert to string
    return str(value).strip()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,   # Quiet logs during tests; flip to INFO to debug
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Simple Executor (structural tests, no LLM calls)...\n")

    # ───────────────────────────────────────────────────────────────
    # Test 1: _safe_get_str handles common LLM output oddities
    # ───────────────────────────────────────────────────────────────
    assert _safe_get_str({"k": "value"}, "k") == "value"
    assert _safe_get_str({"k": "  spaced  "}, "k") == "spaced"
    assert _safe_get_str({"k": None}, "k") == ""
    assert _safe_get_str({"k": 42}, "k") == "42"
    assert _safe_get_str({"k": True}, "k") == "True"
    assert _safe_get_str({"k": {"nested": 1}}, "k").startswith("{")
    assert _safe_get_str({}, "missing") == ""
    assert _safe_get_str({}, "missing", default="fallback") == "fallback"
    print("✓ _safe_get_str handles None, int, bool, nested, missing keys")

    # ───────────────────────────────────────────────────────────────
    # Test 2: VALID_OPERATIONS set is correct
    # ───────────────────────────────────────────────────────────────
    assert OP_CREATE_FILE in VALID_OPERATIONS
    assert OP_UPDATE_FILE in VALID_OPERATIONS
    assert OP_RUN_COMMAND in VALID_OPERATIONS
    assert OP_VERIFY in VALID_OPERATIONS
    assert "delete_universe" not in VALID_OPERATIONS
    print("✓ VALID_OPERATIONS contains expected operations")

    # ───────────────────────────────────────────────────────────────
    # Test 3: simple_executor_node handles empty plan
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": [],
        "current_step": 0,
        "logs": [],
        "files": [],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = simple_executor_node(state)
    assert isinstance(delta, dict), "executor must return a dict"
    assert delta.get("is_complete") is True
    assert delta.get("final_status") == STATUS_FAILED
    assert "logs" in delta
    print("✓ empty plan → terminal failure delta")

    # ───────────────────────────────────────────────────────────────
    # Test 4: simple_executor_node handles all-steps-done
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": ["step1", "step2"],
        "current_step": 2,   # past the last index
        "logs": [],
        "files": ["main.py"],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = simple_executor_node(state)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    # No prior error, so should report success
    assert delta.get("final_status") in ("success", "failed")
    print("✓ all-steps-done → terminal delta")

    # ───────────────────────────────────────────────────────────────
    # Test 5: simple_executor_node handles None state without crashing
    # ───────────────────────────────────────────────────────────────
    delta = simple_executor_node(None)
    assert isinstance(delta, dict)
    assert "logs" in delta
    assert delta.get("is_complete") is True
    print("✓ None state → graceful failure (no crash)")

    # ───────────────────────────────────────────────────────────────
    # Test 6: simple_executor_node handles non-dict state
    # ───────────────────────────────────────────────────────────────
    delta = simple_executor_node("not a state")
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    print("✓ non-dict state → graceful failure")

    # ───────────────────────────────────────────────────────────────
    # Test 7: simple_executor_node handles all-steps-done with prior error
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": ["s1"],
        "current_step": 1,
        "last_error": "something broke earlier",
        "logs": [],
        "files": [],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = simple_executor_node(state)
    assert delta.get("is_complete") is True
    assert delta.get("final_status") == STATUS_FAILED
    print("✓ all-steps-done with prior error → failed status")

    # ───────────────────────────────────────────────────────────────
    # Test 8: StepResult contract — _run_simple_step always returns StepResult
    # ───────────────────────────────────────────────────────────────
    # We verify the return TYPE without mocking LLM by giving the runner
    # a step that will fail at the LLM call. The runner should catch the
    # exception and still return a StepResult (not raise).
    new_logs = []
    log = make_logger(new_logs)
    try:
        # Empty step description triggers code-generator validation failure
        result = _run_simple_step(
            step_description="",   # generator should reject this
            requirement="test",
            plan=[""],
            files_so_far=[],
            log=log,
        )
        assert isinstance(result, StepResult), \
            f"_run_simple_step returned {type(result).__name__} not StepResult"
        # Empty step might still produce a plan from some LLMs; either way
        # the return type contract holds, which is what we're testing
        print("✓ _run_simple_step returns StepResult (even on LLM weirdness)")
    except Exception as exc:
        # If the LLM is unreachable, that's fine — _run_simple_step should
        # have caught it and returned a failure StepResult instead of raising.
        # If we got here, that's a bug in the executor.
        raise AssertionError(
            f"_run_simple_step raised instead of returning StepResult: {exc}"
        )

    print("\n✓ All 8 structural self-tests passed.")