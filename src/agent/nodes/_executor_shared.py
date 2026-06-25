"""
Executor Shared Helpers

Foundation module for the three specialized executors:
- simple_executor.py (Python scripts, basic CLI tools)
- compiled_executor.py (C, C++, Rust, Go, Java)
- web_executor.py (FastAPI, Flask, Streamlit)

Everything in this file is project-type-AGNOSTIC. Logic that differs per
project type lives in the specialized executor files themselves.

Design principles applied:
- Never-Raises: every public function returns a valid state delta, even on
  unexpected input
- DRY: ~70% of executor code lives here, ~30% in each specialized file
- Open/Closed: adding a 4th executor (e.g. data science) means adding one
  new file + one router map entry, with zero changes to this module
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.agent.state import AgentState
from src.tools.code_generator import generate_execution_plan
from src.tools.error_analyzer import analyze_execution_failure
from src.tools.file_manager import create_or_update_file
from src.tools.fix_generator import apply_fixes, generate_fixes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3                    # Total failure attempts before replanning
MAX_FIX_ATTEMPTS_PER_STEP = 2      # LLM-driven fix attempts before raw retry

# ---------------------------------------------------------------------------
# Status constants (canonical vocabulary for final_status)
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"


# ---------------------------------------------------------------------------
# StepResult — what every step runner returns
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """
    Outcome of executing a single plan step.

    All three specialized executors return one of these from their
    `_run_X_step` function. The shared layer then translates the result
    into the appropriate state delta.
    """
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

def dedupe_preserve_order(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving the original order.

    Used to clean the files list — same file can appear multiple times
    if multiple steps write to it. We want unique paths, ordered by
    first appearance.
    """
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_delta(**updates) -> dict:
    """
    Build a LangGraph state-update dict, dropping `...` sentinels.

    Why: lets a caller selectively omit fields by passing `...`, which is
    cleaner than building conditional dicts.

    Example:
        return make_delta(
            logs=new_logs,
            files=merged_files,
            final_status=status if is_done else ...,  # omit when not done
        )
    """
    return {k: v for k, v in updates.items() if v is not ...}


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def extract_state_basics(state: AgentState) -> dict:
    """
    Pull common fields from state into a flat dict with safe defaults.

    Every specialized executor calls this first to normalize state into
    a predictable shape. Defensive — handles missing fields, wrong types,
    None values, and even non-dict state.
    """
    if not isinstance(state, dict):
        logger.warning("Executor received non-dict state (%s); using defaults",
                       type(state).__name__)
        state = {}

    return {
        "plan":             _safe_list(state.get("plan")),
        "current_step":     _safe_int(state.get("current_step"), default=0),
        "existing_logs":    _safe_list(state.get("logs")),
        "existing_files":   _safe_list(state.get("files")),
        "retry_count":      _safe_int(state.get("retry_count"), default=0),
        "prior_last_error": state.get("last_error"),
        "requirement":      _safe_str(state.get("requirement")),
    }


def _safe_list(value: Any) -> list:
    """Coerce to list, rejecting strings (which would split into characters)."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []  # None, str, dict, int, etc. all become empty list


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce to int, handling strings like '5' and falling back on garbage."""
    if isinstance(value, bool):
        # bool is a subclass of int; explicit handling avoids surprises
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return default
    return default


def _safe_str(value: Any) -> str:
    """Coerce to stripped string, returning empty string for non-string types."""
    if isinstance(value, str):
        return value.strip()
    return ""


def make_logger(new_logs: List[str]) -> Callable:
    """
    Create a `log()` closure that both prints to stdout (via Python logging)
    AND appends to a list for inclusion in the state delta.

    Why: executor logs need to be visible in real-time (for debugging) AND
    propagated to LangGraph state (for the UI/summary). One call, both effects.

    Usage in an executor:
        new_logs = []
        log = make_logger(new_logs)
        log("Starting step 1")  # prints AND appends to new_logs
    """
    def log(msg: str, level: int = logging.INFO) -> None:
        logger.log(level, msg)
        new_logs.append(msg)
    return log


# ---------------------------------------------------------------------------
# Code generation wrapper
# ---------------------------------------------------------------------------

def generate_plan_for_step(
    step_description: str,
    requirement: str,
    plan_overview: List[str],
    files_so_far: List[str],
) -> Dict[str, Any]:
    """
    Call the code generator with full context.

    Wraps the raw `generate_execution_plan` so all executors pass context
    consistently. Centralizes the "what to send to the LLM" decision.
    """
    return generate_execution_plan(
        step_description=step_description,
        requirement=requirement,
        plan_overview=plan_overview,
        files_so_far=files_so_far,
    )


# ---------------------------------------------------------------------------
# File writing helper
# ---------------------------------------------------------------------------

def write_file_if_needed(
    file_path: str,
    file_content: str,
    result: StepResult,
    log: Callable,
) -> bool:
    """
    Write a file via file_manager. Updates `result` with files_created or
    error info. Returns True on success, False on failure.

    Why a helper: all three executors do this identically. The only thing
    that varies is what happens AFTER the file is written (compile? probe?
    just run?), and that's what specializes them.
    """
    if not file_path:
        log("[file] skipped (no file_path)")
        return True

    log(f"[file] writing {file_path}")
    ok, err, created = create_or_update_file(file_path, file_content)
    if not ok:
        log(f"[file] ✗ failed: {err}", level=logging.ERROR)
        result.error_message = f"File creation failed: {err}"
        result.stderr = err
        return False

    result.files_created.extend(created)
    log(f"[file] ✓ wrote {created}")
    return True


# ---------------------------------------------------------------------------
# Top-level guards (called at the start of every executor_node)
# ---------------------------------------------------------------------------

def check_empty_plan(
    state_basics: dict,
    new_logs: List[str],
    log: Callable,
) -> Optional[dict]:
    """
    If the plan is empty, return a "nothing to do" terminal delta.
    Otherwise return None and let the caller continue.
    """
    if state_basics["plan"]:
        return None

    log("No plan; nothing to execute.")
    return make_delta(
        logs=state_basics["existing_logs"] + new_logs,
        files=state_basics["existing_files"],
        is_complete=True,
        current_step=0,
        retry_count=0,
        last_error="No plan provided",
        final_status=STATUS_FAILED,
    )


def check_all_steps_done(
    state_basics: dict,
    new_logs: List[str],
    log: Callable,
) -> Optional[dict]:
    """
    If all plan steps have been processed, return the final terminal delta.
    Otherwise return None.

    final_status reflects whether anything errored along the way:
    - any prior_last_error → "failed"
    - clean run → "success"
    """
    if state_basics["current_step"] < len(state_basics["plan"]):
        return None

    status = (
        STATUS_FAILED if state_basics["prior_last_error"] else STATUS_SUCCESS
    )
    log(f"All {len(state_basics['plan'])} steps processed. Status: {status}")

    return make_delta(
        logs=state_basics["existing_logs"] + new_logs,
        files=dedupe_preserve_order(state_basics["existing_files"]),
        is_complete=True,
        current_step=state_basics["current_step"],
        retry_count=0,
        last_error=state_basics["prior_last_error"],
        final_status=status,
    )


# ---------------------------------------------------------------------------
# Success path — when a step completes successfully
# ---------------------------------------------------------------------------

def build_success_delta(
    state_basics: dict,
    result: StepResult,
    new_logs: List[str],
    log: Callable,
) -> dict:
    """
    After a step succeeds: advance step counter, dedupe files, and decide
    whether to terminate (if last step) or continue.
    """
    merged_files = dedupe_preserve_order(
        state_basics["existing_files"] + result.files_created
    )
    new_step = state_basics["current_step"] + 1
    is_done = new_step >= len(state_basics["plan"])

    verification_note = (
        f" [verified: {result.verification_summary}]"
        if result.verification_summary else ""
    )
    log(
        f"✓ Step {state_basics['current_step'] + 1} done.{verification_note} "
        f"Files: {result.files_created or 'none'}"
    )

    if is_done:
        final_status = (
            STATUS_SUCCESS if not state_basics["prior_last_error"]
            else STATUS_FAILED
        )
        log(f"All steps complete. final_status={final_status}")
        return make_delta(
            logs=state_basics["existing_logs"] + new_logs,
            files=merged_files,
            current_step=new_step,
            retry_count=0,
            last_error=state_basics["prior_last_error"],
            plan_feedback=None,
            is_complete=True,
            final_status=final_status,
        )

    # Mid-plan success: clear error, advance, continue
    return make_delta(
        logs=state_basics["existing_logs"] + new_logs,
        files=merged_files,
        current_step=new_step,
        retry_count=0,
        last_error=None,
        plan_feedback=None,
        is_complete=False,
    )


# ---------------------------------------------------------------------------
# Failure path — analyze, attempt fix, retry or escalate
# ---------------------------------------------------------------------------

def handle_step_failure(
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
    """
    Diagnose a failure and decide what to do next.

    Decision tree:
      1. Run error analyzer to classify the failure
      2. If recoverable AND we haven't burned our fix budget → try a fix
      3. If max retries reached → escalate to planner with feedback
      4. Otherwise → retry the same step

    This is identical across all three executors because error recovery
    doesn't care about project type — it cares about error type.
    """
    retry_count += 1
    logs = list(base_logs)
    logs.append(f"--- ERROR RECOVERY (attempt {retry_count}/{MAX_RETRIES}) ---")
    logger.info("Error recovery attempt %d/%d", retry_count, MAX_RETRIES)

    # Step 1: Analyze the error
    error_analysis: Optional[Dict[str, Any]] = None
    if stdout or stderr:
        try:
            error_analysis = analyze_execution_failure(stdout, stderr, command)
            logs.append(
                f"Error Analyzer: type={error_analysis.get('error_type')} "
                f"recoverable={error_analysis.get('is_recoverable')}"
            )
        except Exception as exc:
            logger.exception("Error analyzer crashed")
            logs.append(f"Error Analyzer crashed: {exc}")

    # Step 2: Try a fix if safe
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
            fix_ok, applied, _ = apply_fixes(fix_plan)
            for desc in applied:
                logs.append(f"  applied: {desc}")

            if fix_ok and applied:
                logs.append(f"⚠ Retrying step {current_step + 1} with fixes")
                return make_delta(
                    logs=logs,
                    files=base_files,
                    current_step=current_step,
                    retry_count=retry_count,
                    last_error=error_msg,
                    is_complete=False,
                )
        except Exception as exc:
            logger.exception("Fix generator crashed")
            logs.append(f"Fix Generator crashed: {exc}")

    # Step 3: Escalate or retry
    if retry_count >= MAX_RETRIES:
        feedback = build_plan_feedback(step_description, error_msg, error_analysis)
        logs.append(
            f"✗ Step {current_step + 1} failed after {MAX_RETRIES} attempts. "
            "Escalating to planner."
        )
        return make_delta(
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
            f"⚠ Step {current_step + 1} failed (retry {retry_count}/{MAX_RETRIES})"
        )
    return make_delta(
        logs=logs,
        files=base_files,
        current_step=current_step,
        retry_count=retry_count,
        last_error=error_msg,
        is_complete=False,
    )


def build_plan_feedback(
    step_description: str,
    error_msg: str,
    error_analysis: Optional[Dict[str, Any]],
) -> str:
    """
    Compose a structured message for the planner to replan from.

    Includes error type, severity, root cause, and suggested fix when
    available — gives the planner concrete signal to work with.
    """
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
# Self-test (minimal — full integration tests live in the specialized files)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing shared executor helpers...\n")

    # ───────────────────────────────────────────────────────────────
    # Test 1: dedupe_preserve_order
    # ───────────────────────────────────────────────────────────────
    assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedupe_preserve_order([]) == []
    assert dedupe_preserve_order(["x", "", "x"]) == ["x"]
    assert dedupe_preserve_order(["only"]) == ["only"]
    print("✓ dedupe_preserve_order")

    # ───────────────────────────────────────────────────────────────
    # Test 2: make_delta drops sentinels but keeps None
    # ───────────────────────────────────────────────────────────────
    delta = make_delta(a=1, b=..., c=None, d="value")
    assert delta == {"a": 1, "c": None, "d": "value"}, f"got {delta}"
    assert make_delta() == {}
    print("✓ make_delta")

    # ───────────────────────────────────────────────────────────────
    # Test 3: extract_state_basics (defensive against all input types)
    # ───────────────────────────────────────────────────────────────
    # 3a. None state → all defaults
    s = extract_state_basics(None)
    assert s["plan"] == []
    assert s["current_step"] == 0
    assert s["existing_logs"] == []
    assert s["existing_files"] == []
    assert s["retry_count"] == 0
    assert s["prior_last_error"] is None
    assert s["requirement"] == ""

    # 3b. Empty dict → all defaults
    s = extract_state_basics({})
    assert s["plan"] == []
    assert s["current_step"] == 0

    # 3c. Wrong type for plan (string) → empty list, NOT character split
    s = extract_state_basics({"plan": "not a list", "current_step": "5"})
    assert s["plan"] == [], f"expected [], got {s['plan']}"
    assert s["current_step"] == 5, f"expected 5, got {s['current_step']}"

    # 3d. Garbage current_step → default 0
    s = extract_state_basics({"current_step": "garbage"})
    assert s["current_step"] == 0

    # 3e. Float current_step → int
    s = extract_state_basics({"current_step": 3.7})
    assert s["current_step"] == 3

    # 3f. Valid full state passes through
    s = extract_state_basics({
        "plan": ["step1", "step2"],
        "current_step": 1,
        "logs": ["log1"],
        "files": ["main.py"],
        "retry_count": 2,
        "last_error": "something",
        "requirement": "  build a thing  ",
    })
    assert s["plan"] == ["step1", "step2"]
    assert s["current_step"] == 1
    assert s["existing_logs"] == ["log1"]
    assert s["existing_files"] == ["main.py"]
    assert s["retry_count"] == 2
    assert s["prior_last_error"] == "something"
    assert s["requirement"] == "build a thing"   # stripped

    # 3g. Non-dict state (string) → defaults
    s = extract_state_basics("not a state")
    assert s["plan"] == []
    print("✓ extract_state_basics (handles 7 edge cases)")

    # ───────────────────────────────────────────────────────────────
    # Test 4: make_logger captures AND prints
    # ───────────────────────────────────────────────────────────────
    captured = []
    log = make_logger(captured)
    log("hello")
    log("world")
    assert captured == ["hello", "world"], f"got {captured}"
    print("✓ make_logger")

    # ───────────────────────────────────────────────────────────────
    # Test 5: StepResult — no shared mutable state between instances
    # ───────────────────────────────────────────────────────────────
    r1 = StepResult()
    assert r1.success is False
    assert r1.files_created == []
    assert r1.plan == {}

    r1.files_created.append("main.py")
    r2 = StepResult()
    assert r2.files_created == [], \
        f"StepResult instances share mutable state! got {r2.files_created}"
    print("✓ StepResult (no shared mutable state)")

    # ───────────────────────────────────────────────────────────────
    # Test 6: check_empty_plan
    # ───────────────────────────────────────────────────────────────
    # Empty plan → returns terminal failure delta
    new_logs = []
    log = make_logger(new_logs)
    s = extract_state_basics({"plan": []})
    delta = check_empty_plan(s, new_logs, log)
    assert delta is not None
    assert delta["is_complete"] is True
    assert delta["final_status"] == STATUS_FAILED

    # Non-empty plan → returns None (continue)
    new_logs = []
    log = make_logger(new_logs)
    s = extract_state_basics({"plan": ["step1"]})
    delta = check_empty_plan(s, new_logs, log)
    assert delta is None
    print("✓ check_empty_plan")

    # ───────────────────────────────────────────────────────────────
    # Test 7: check_all_steps_done
    # ───────────────────────────────────────────────────────────────
    # All steps done, no errors → success
    new_logs = []
    log = make_logger(new_logs)
    s = extract_state_basics({
        "plan": ["s1", "s2"],
        "current_step": 2,  # past the last index
    })
    delta = check_all_steps_done(s, new_logs, log)
    assert delta is not None
    assert delta["is_complete"] is True
    assert delta["final_status"] == STATUS_SUCCESS

    # All steps done, but had an error → failed
    new_logs = []
    log = make_logger(new_logs)
    s = extract_state_basics({
        "plan": ["s1", "s2"],
        "current_step": 2,
        "last_error": "something broke",
    })
    delta = check_all_steps_done(s, new_logs, log)
    assert delta is not None
    assert delta["final_status"] == STATUS_FAILED

    # Mid-plan → returns None
    new_logs = []
    log = make_logger(new_logs)
    s = extract_state_basics({"plan": ["s1", "s2"], "current_step": 1})
    delta = check_all_steps_done(s, new_logs, log)
    assert delta is None
    print("✓ check_all_steps_done")

    # ───────────────────────────────────────────────────────────────
    # Test 8: build_plan_feedback structure
    # ───────────────────────────────────────────────────────────────
    fb = build_plan_feedback(
        "Write main.py",
        "command failed",
        {"error_type": "SyntaxError", "severity": "high",
         "root_cause": "missing semicolon", "suggested_fix": "add semicolon"},
    )
    assert "Write main.py" in fb
    assert "SyntaxError" in fb
    assert "add semicolon" in fb
    print("✓ build_plan_feedback")

    print("\n✓ All 8 self-tests passed. Shared helpers are bulletproof.")