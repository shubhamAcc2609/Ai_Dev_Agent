"""
Fix Generator

Generates corrective actions for execution failures and applies them via the
existing file_manager / execution_manager modules.

Two-stage design:
    1. Generate — LLM produces a structured fix plan; deterministic fallback
       handles common cases (missing modules, etc.) when the LLM is unavailable.
    2. Apply — dispatch each fix to a typed handler with safe argument
       construction. Dangerous commands and unsafe pip names are rejected
       before reaching the shell.

Auto-retry loop uses error fingerprinting to detect "no progress" cycles
and escalates via an optional callback rather than spinning indefinitely.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm
from src.tools.error_analyzer import analyze_execution_failure
from src.tools.execution_manager import execute_command
from src.tools.file_manager import create_or_update_file, delete_file
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FIXES_PER_PLAN = 10
DEFAULT_PRIORITY = 5
DEFAULT_CONFIDENCE = 50
INSTALL_TIMEOUT_SEC = 120


class FixType(str, Enum):
    INSTALL_DEPENDENCY = "install_dependency"
    MODIFY_FILE = "modify_file"
    CREATE_FILE = "create_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"


# PEP 508-ish package name with optional version pin and extras.
# Matches `flask`, `requests==2.31.0`, `uvicorn[standard]`.
PIP_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._\-]*"
    r"(\[[A-Za-z0-9_,\-]+\])?"
    r"(==[\w\.\-\+]+)?$"
)

# Patterns we refuse to execute even if the LLM suggests them.
DANGEROUS_RE = re.compile(
    r"\b(?:rm\s+-rf\s+/|sudo\s|mkfs|shutdown|reboot|"
    r"format\s+c:|del\s+/[fsq]+(?:\s+/[fsq]+)*\s+c:\\)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Fix:
    type: str
    target: str
    action: str = ""
    content: Optional[str] = None
    priority: int = DEFAULT_PRIORITY

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Fix"]:
        """Parse a fix dict from LLM output. Returns None on invalid input."""
        if not isinstance(data, dict):
            return None

        fix_type = str(data.get("type", "")).strip()
        if fix_type not in {ft.value for ft in FixType}:
            return None

        try:
            priority = max(1, min(10, int(data.get("priority", DEFAULT_PRIORITY))))
        except (TypeError, ValueError):
            priority = DEFAULT_PRIORITY

        return cls(
            type=fix_type,
            target=str(data.get("target", "")).strip(),
            action=str(data.get("action", "")).strip(),
            content=data.get("content"),
            priority=priority,
        )


@dataclass
class FixPlan:
    fixes: List[Fix] = field(default_factory=list)
    verification_command: str = ""
    rollback_steps: List[str] = field(default_factory=list)
    explanation: str = ""
    confidence: int = DEFAULT_CONFIDENCE

    def to_dict(self) -> Dict:
        return {
            "fixes": [
                {"type": f.type, "target": f.target, "action": f.action,
                 "content": f.content, "priority": f.priority}
                for f in self.fixes
            ],
            "verification_command": self.verification_command,
            "rollback_steps": self.rollback_steps,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert fix generator for an autonomous Software Development Agent.

Generate fixes for execution failures. Respond with ONLY one JSON object — no
markdown fences, no prose, no <think> tags.

Schema:
{
  "fixes": [
    {
      "type": "install_dependency | modify_file | create_file | delete_file | run_command",
      "target": "<package name | project-relative file path | short command label>",
      "action": "<exact command for run_command/install; short description otherwise>",
      "content": "<COMPLETE file content for modify_file/create_file, else null>",
      "priority": <integer 1-10, higher = apply first>
    }
  ],
  "verification_command": "<command that should succeed after fixes>",
  "rollback_steps": ["<step 1>", "<step 2>"],
  "explanation": "<why these fixes address the root cause>",
  "confidence": <integer 0-100>
}

Rules:
- File paths must be project-relative (no "..", no leading "/", no drive letters).
- For modify_file / create_file, content must be the COMPLETE file body.
- For install_dependency, target is the bare package name (e.g. "flask",
  "requests==2.31.0"). Do not include "pip install" in target.
- run_command is for builds, tests, git, or installs that aren't single packages.
- Never use sudo, rm -rf /, or other destructive shell tricks.
-When wrapping risky code in try/except, the except block must PRINT a 
descriptive message (e.g., "Error: division by zero"). Never use silent 
`pass` — the user must see what was caught.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_fixes(
    error_analysis: Dict,
    code_context: str = "",
    file_path: str = "",
) -> Dict:
    """Generate a validated fix plan for the given error analysis."""
    if not isinstance(error_analysis, dict) or not error_analysis:
        raise ValueError("error_analysis must be a non-empty dict")

    logger.info(
        "Generating fixes (type=%s severity=%s)",
        error_analysis.get("error_type"),
        error_analysis.get("severity"),
    )

    fallback = _build_fallback_plan(error_analysis)

    llm_plan = _call_llm(error_analysis, code_context, file_path)
    if llm_plan is None:
        logger.info("LLM unavailable; using deterministic fallback")
        return fallback.to_dict()

    plan = _merge_with_fallback(llm_plan, fallback)
    logger.info(
        "Fix plan ready: %d fix(es), confidence=%d%%",
        len(plan.fixes), plan.confidence,
    )
    return plan.to_dict()


def apply_fixes(
    fix_plan: Dict,
    dry_run: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Apply fixes in priority order.

    Returns:
        (overall_success, applied_descriptions, error_descriptions)
    """
    fixes = _parse_fix_list(fix_plan)
    if not fixes:
        return False, [], ["No valid fixes in plan"]

    logger.info("Applying %d fix(es)%s",
                len(fixes), " [dry-run]" if dry_run else "")

    applied: List[str] = []
    errors: List[str] = []

    for i, fix in enumerate(fixes, 1):
        logger.info("Fix %d/%d: type=%s target=%r",
                    i, len(fixes), fix.type, fix.target)
        ok, message = _dispatch(fix, dry_run)
        (applied if ok else errors).append(message)

    success = bool(applied) and not errors
    logger.info("Apply summary: %d ok, %d failed", len(applied), len(errors))
    return success, applied, errors


def verify_fix(
    verification_command: str,
    original_command: str,
) -> Tuple[bool, str]:
    """
    Confirm the fix worked.

    Runs the verification command first (cheap, targeted signal). Falls back
    to re-running the original failing command.
    """
    if verification_command:
        ok, stdout, stderr = execute_command(verification_command)
        if not ok:
            return False, stderr or stdout

    ok, stdout, stderr = execute_command(original_command)
    return ok, stdout if ok else (stderr or stdout)


def auto_fix_and_retry(
    command: str,
    max_retries: int = 3,
    code_context: str = "",
    file_path: str = "",
    on_escalation: Optional[Callable[[Dict], None]] = None,
) -> Tuple[bool, str, str, List[Dict]]:
    """
    Run a command and auto-fix failures up to `max_retries` times.

    Loop detection: if the same error fingerprint appears twice consecutively,
    we give up rather than spin. Escalation callback fires on terminal cases.
    """
    history: List[Dict] = []
    last_fingerprint = ""
    last_stdout = last_stderr = ""

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt %d/%d", attempt, max_retries)
        success, last_stdout, last_stderr = execute_command(command)
        if success:
            return True, last_stdout, last_stderr, history

        analysis = analyze_execution_failure(last_stdout, last_stderr, command)
        fingerprint = _fingerprint(analysis)

        if fingerprint == last_fingerprint:
            logger.warning("No progress — same error twice; escalating")
            _escalate(on_escalation, analysis, history, "no_progress")
            return False, last_stdout, last_stderr, history

        if not analysis.get("is_recoverable", False):
            logger.info("Error marked non-recoverable; escalating")
            _escalate(on_escalation, analysis, history, "non_recoverable")
            return False, last_stdout, last_stderr, history

        plan = generate_fixes(analysis, code_context, file_path)
        ok, applied, errors = apply_fixes(plan)

        history.append({
            "attempt": attempt,
            "error_analysis": analysis,
            "fix_plan": plan,
            "applied": applied,
            "errors": errors,
            "fingerprint": fingerprint,
        })

        if not ok:
            _escalate(on_escalation, analysis, history, "apply_failed")
            return False, last_stdout, last_stderr, history

        last_fingerprint = fingerprint

    logger.warning("Exhausted max_retries=%d", max_retries)
    final_analysis = history[-1]["error_analysis"] if history else {}
    _escalate(on_escalation, final_analysis, history, "max_retries")
    return False, last_stdout, last_stderr, history


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _call_llm(
    error_analysis: Dict,
    code_context: str,
    file_path: str,
) -> Optional[Dict]:
    """
    Ask the LLM for a fix plan. Returns the parsed JSON object, or None if
    the LLM call or parsing failed.
    """
    user_text = (
        "Generate fixes for this error:\n\n"
        f"Error Type: {error_analysis.get('error_type', 'Unknown')}\n"
        f"Error Message: {error_analysis.get('error_message', '')}\n"
        f"Root Cause: {error_analysis.get('root_cause', '')}\n"
        f"Severity: {error_analysis.get('severity', 'major')}\n"
        f"Affected Component: {error_analysis.get('affected_component', 'Unknown')}\n"
        f"File Path: {file_path}\n\n"
        f"Code Context:\n{code_context[:4000]}\n\n"
        f"Suggested Fix Direction: {error_analysis.get('suggested_fix', '')}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ])
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None

    text = getattr(response, "content", "") or ""
    if not text.strip():
        return None

    try:
        parsed = extract_json_object(text)
    except ValueError as exc:
        logger.warning("Failed to parse LLM fix plan: %s", exc)
        return None

    return parsed if isinstance(parsed, dict) else None


def _merge_with_fallback(raw: Dict, fallback: FixPlan) -> FixPlan:
    """
    Build a FixPlan from LLM output, preferring LLM values but falling back
    to defaults when fields are missing or invalid.
    """
    raw_fixes = raw.get("fixes") or []
    if not isinstance(raw_fixes, list):
        return fallback

    parsed = [Fix.from_dict(f) for f in raw_fixes[:MAX_FIXES_PER_PLAN]]
    fixes = [f for f in parsed if f is not None]
    if not fixes:
        return fallback

    try:
        confidence = max(0, min(100, int(raw.get("confidence", fallback.confidence))))
    except (TypeError, ValueError):
        confidence = fallback.confidence

    rollback = raw.get("rollback_steps") or fallback.rollback_steps
    if not isinstance(rollback, list):
        rollback = [str(rollback)]

    return FixPlan(
        fixes=fixes,
        verification_command=str(
            raw.get("verification_command") or fallback.verification_command
        ),
        rollback_steps=[str(s) for s in rollback],
        explanation=str(raw.get("explanation") or fallback.explanation),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fix dispatch — each handler validates its own arguments
# ---------------------------------------------------------------------------

def _parse_fix_list(fix_plan: Dict) -> List[Fix]:
    """Extract and prioritize fixes from a plan dict."""
    if not isinstance(fix_plan, dict):
        return []
    raw_fixes = fix_plan.get("fixes", [])
    if not isinstance(raw_fixes, list):
        return []
    fixes = [Fix.from_dict(f) for f in raw_fixes]
    fixes = [f for f in fixes if f is not None]
    fixes.sort(key=lambda f: f.priority, reverse=True)
    return fixes


def _dispatch(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    """Route a fix to the right handler. Each handler returns (ok, message)."""
    handlers = {
        FixType.INSTALL_DEPENDENCY.value: _install_dependency,
        FixType.MODIFY_FILE.value: lambda f, d: _write_file(f, d, "modify"),
        FixType.CREATE_FILE.value: lambda f, d: _write_file(f, d, "create"),
        FixType.DELETE_FILE.value: _delete_file,
        FixType.RUN_COMMAND.value: _run_command,
    }
    handler = handlers.get(fix.type)
    if handler is None:
        return False, f"Unknown fix type: {fix.type}"
    return handler(fix, dry_run)


def _install_dependency(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    pkg = fix.target.strip()
    if not pkg:
        return False, "install_dependency requires a non-empty target"
    if not PIP_NAME_RE.match(pkg):
        return False, f"Unsafe pip package spec rejected: {pkg!r}"

    cmd = f"{_python_exe()} -m pip install {pkg}"
    if dry_run:
        return True, f"[dry-run] would run: {cmd}"

    ok, _, stderr = execute_command(cmd, timeout=INSTALL_TIMEOUT_SEC)
    if ok:
        return True, f"Installed dependency: {pkg}"
    return False, f"Failed to install {pkg}: {stderr.strip()[:200]}"


def _write_file(fix: Fix, dry_run: bool, verb: str) -> Tuple[bool, str]:
    if not fix.target:
        return False, f"{verb}_file requires a target path"
    if not isinstance(fix.content, str):
        return False, f"{verb}_file requires string content"

    if dry_run:
        return True, (
            f"[dry-run] would {verb} {fix.target} ({len(fix.content)} chars)"
        )

    ok, err, files = create_or_update_file(fix.target, fix.content)
    if ok:
        return True, f"{verb.title()}d file: {files[0] if files else fix.target}"
    return False, f"Failed to {verb} {fix.target}: {err}"


def _delete_file(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    if not fix.target:
        return False, "delete_file requires a target path"

    if dry_run:
        return True, f"[dry-run] would delete: {fix.target}"

    ok, err = delete_file(fix.target)
    if ok:
        return True, f"Deleted file: {fix.target}"
    return False, f"Failed to delete {fix.target}: {err}"


def _run_command(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    cmd = fix.action or fix.target
    if not cmd:
        return False, "run_command requires an action or target"

    if DANGEROUS_RE.search(cmd):
        return False, f"Refused dangerous command: {cmd!r}"

    if dry_run:
        return True, f"[dry-run] would run: {cmd}"

    ok, _, stderr = execute_command(cmd)
    if ok:
        return True, f"Executed: {cmd}"
    return False, f"Command failed: {cmd} -- {stderr.strip()[:200]}"


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _build_fallback_plan(error_analysis: Dict) -> FixPlan:
    """
    Build a deterministic fix plan for common errors without calling the LLM.
    Used when the LLM is unavailable or its output is unusable.
    """
    error_type = error_analysis.get("error_type", "UnknownError")
    error_message = error_analysis.get("error_message", "")

    # ModuleNotFoundError / ImportError → suggest pip install
    if error_type in ("ModuleNotFoundError", "ImportError"):
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        if match:
            module = match.group(1).split(".")[0]
            if PIP_NAME_RE.match(module):
                return FixPlan(
                    fixes=[Fix(
                        type=FixType.INSTALL_DEPENDENCY.value,
                        target=module,
                        action=f"pip install {module}",
                        priority=10,
                    )],
                    verification_command=f'{_python_exe()} -c "import {module}"',
                    rollback_steps=["Uninstall added dependency if needed"],
                    explanation=f"Install missing module: {module}",
                    confidence=75,
                )

    # For other error types we don't have safe deterministic actions.
    # Return an empty plan with low confidence so the LLM (or caller) knows.
    confidence_by_type = {
        "DependencyError": 15,
        "PermissionError": 10,
        "SyntaxError": 10,
        "IndentationError": 10,
        "PortInUseError": 20,
    }

    return FixPlan(
        fixes=[],
        rollback_steps=["No deterministic fallback available"],
        explanation=f"No safe automatic fallback for {error_type}",
        confidence=confidence_by_type.get(error_type, 30),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_exe() -> str:
    """Return the current Python executable, quoted if it contains spaces."""
    exe = sys.executable or "python"
    return f'"{exe}"' if " " in exe else exe


def _fingerprint(analysis: Dict) -> str:
    """Stable short hash of an error for loop detection."""
    raw = f"{analysis.get('error_type', '')}|{analysis.get('error_message', '')}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _escalate(
    callback: Optional[Callable[[Dict], None]],
    analysis: Dict,
    history: List[Dict],
    reason: str,
) -> None:
    """Fire the escalation callback if one was registered."""
    if callback is None:
        logger.info("Escalation (reason=%s) — no handler registered", reason)
        return
    callback({
        "reason": reason,
        "error_analysis": analysis,
        "fix_history": history,
    })