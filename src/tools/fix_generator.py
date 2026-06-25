"""
Fix Generator Module

Responsible for:
- Generating fixes from Error Analyzer output (LLM + deterministic fallback)
- Safely applying corrective actions (dependency install, file create/update/delete,
  shell commands) via the existing file_manager / execution_manager modules
- Verifying fixes by re-running the original failing command
- Driving an auto-retry loop with loop-detection and escalation hooks
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from src.agent.config import llm
from src.tools.error_analyzer import analyze_execution_failure
from src.tools.execution_manager import execute_command
from src.tools.file_manager import create_or_update_file, delete_file
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

class FixType(str, Enum):
    INSTALL_DEPENDENCY = "install_dependency"
    MODIFY_FILE        = "modify_file"
    CREATE_FILE        = "create_file"
    DELETE_FILE        = "delete_file"
    RUN_COMMAND        = "run_command"


ALLOWED_FIX_TYPES = {ft.value for ft in FixType}

DEFAULT_PRIORITY = 5
DEFAULT_CONFIDENCE = 50
MAX_FIXES_PER_PLAN = 10

# Package-name regex (PEP 508-ish) for safe `pip install` arg construction
PIP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(\[[A-Za-z0-9_,\-]+\])?(==[\w\.\-\+]+)?$")

# Tokens we refuse to execute even if the LLM suggests them
DANGEROUS_TOKENS = (
    "rm -rf /", "sudo ", "mkfs", "shutdown", "reboot",
    ":(){:|:&};:", "> /dev/sd", "format c:", "del /f /s /q c:",
)


@dataclass
class Fix:
    type: str
    target: str
    action: str = ""
    content: Optional[str] = None
    priority: int = DEFAULT_PRIORITY

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Fix"]:
        if not isinstance(data, dict):
            return None
        fix_type = str(data.get("type", "")).strip()
        if fix_type not in ALLOWED_FIX_TYPES:
            logger.warning("Skipping fix with unknown type: %r", fix_type)
            return None
        try:
            priority = int(data.get("priority", DEFAULT_PRIORITY))
        except (TypeError, ValueError):
            priority = DEFAULT_PRIORITY
        return cls(
            type=fix_type,
            target=str(data.get("target", "")).strip(),
            action=str(data.get("action", "")).strip(),
            content=data.get("content"),
            priority=max(1, min(10, priority)),
        )


@dataclass
class FixPlan:
    fixes: List[Fix] = field(default_factory=list)
    verification_command: str = ""
    rollback_steps: List[str] = field(default_factory=list)
    explanation: str = ""
    confidence: int = DEFAULT_CONFIDENCE

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["fixes"] = [asdict(f) for f in self.fixes]
        return d


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

FIX_GENERATOR_SYSTEM_PROMPT = """\
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

RULES:
- File paths MUST be project-relative (no "..", no leading "/", no drive letters).
- For modify_file / create_file you MUST return the COMPLETE file content.
- For install_dependency, "target" is the bare package name (e.g. "flask",
  "requests==2.31.0"). Do NOT include "pip install" in target.
- run_command is for builds, tests, git, or installs that aren't single packages.
- Never use sudo, rm -rf /, or other destructive shell tricks.
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

    fallback = _generate_fallback_fixes(error_analysis, file_path)

    try:
        llm_plan_raw = _invoke_llm(error_analysis, code_context, file_path)
    except Exception as exc:
        logger.warning("LLM fix generation failed (%s); using fallback", exc)
        return fallback.to_dict()

    plan = _validate_fix_plan(llm_plan_raw, fallback)
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

    Args:
        fix_plan: Dict returned by `generate_fixes`.
        dry_run: If True, log what would happen but don't touch disk/exec.

    Returns:
        (overall_success, applied_descriptions, error_descriptions)
    """
    raw_fixes = fix_plan.get("fixes", []) if isinstance(fix_plan, dict) else []
    fixes = [Fix.from_dict(f) for f in raw_fixes]
    fixes = [f for f in fixes if f is not None]
    fixes.sort(key=lambda f: f.priority, reverse=True)

    logger.info("Applying %d fix(es)%s", len(fixes), " [dry-run]" if dry_run else "")

    applied: List[str] = []
    errors: List[str] = []

    for i, fix in enumerate(fixes, 1):
        logger.info("Fix %d/%d: type=%s target=%r", i, len(fixes), fix.type, fix.target)
        try:
            ok, msg = _dispatch_fix(fix, dry_run=dry_run)
        except Exception as exc:  # last-resort guard
            logger.exception("Unhandled error applying fix")
            errors.append(f"{fix.type}({fix.target}) raised: {exc}")
            continue

        if ok:
            applied.append(msg)
        else:
            errors.append(msg)

    overall = len(errors) == 0 and len(applied) > 0
    logger.info("Apply summary: %d ok, %d failed", len(applied), len(errors))
    return overall, applied, errors


def verify_fix(
    verification_command: str,
    original_command: str,
) -> Tuple[bool, str]:
    """
    Verify a fix.

    Strategy:
    1. Run the verification command first (cheap, specific signal).
    2. If it passes (or none given), re-run the original failing command.
    """
    if verification_command:
        logger.info("Running verification: %r", verification_command)
        ok, stdout, stderr = execute_command(verification_command)
        if not ok:
            logger.info("Verification command failed")
            return False, stderr or stdout

    logger.info("Re-running original command: %r", original_command)
    ok, stdout, stderr = execute_command(original_command)
    if ok:
        logger.info("Original command now succeeds")
        return True, stdout
    return False, stderr or stdout


def auto_fix_and_retry(
    command: str,
    max_retries: int = 3,
    code_context: str = "",
    file_path: str = "",
    on_escalation: Optional[Callable[[Dict], None]] = None,
) -> Tuple[bool, str, str, List[Dict]]:
    """
    Run a command and auto-fix failures up to `max_retries` times.

    Includes:
    - Loop detection: same error fingerprint twice in a row → give up.
    - Escalation hook: callback fired when fix is impossible.
    """
    logger.info("auto_fix_and_retry: %r (max_retries=%d)", command, max_retries)

    fix_history: List[Dict] = []
    seen_fingerprints: List[str] = []
    last_stdout = last_stderr = ""

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt %d/%d", attempt, max_retries)
        success, last_stdout, last_stderr = execute_command(command)
        if success:
            logger.info("Command succeeded on attempt %d", attempt)
            return True, last_stdout, last_stderr, fix_history

        error_analysis = analyze_execution_failure(last_stdout, last_stderr, command)
        fingerprint = _fingerprint_error(error_analysis)

        # Loop detection: same root cause twice → no progress
        if seen_fingerprints and seen_fingerprints[-1] == fingerprint:
            logger.warning("Same error fingerprint as previous attempt — aborting")
            _maybe_escalate(on_escalation, error_analysis, fix_history,
                            reason="no_progress")
            return False, last_stdout, last_stderr, fix_history
        seen_fingerprints.append(fingerprint)

        if not error_analysis.get("is_recoverable", False):
            logger.info("Error marked non-recoverable; escalating")
            _maybe_escalate(on_escalation, error_analysis, fix_history,
                            reason="non_recoverable")
            return False, last_stdout, last_stderr, fix_history

        try:
            fix_plan = generate_fixes(error_analysis, code_context, file_path)
        except Exception as exc:
            logger.error("Fix generation failed: %s", exc)
            return False, last_stdout, last_stderr, fix_history

        apply_success, applied, errors = apply_fixes(fix_plan)
        fix_history.append({
            "attempt": attempt,
            "error_analysis": error_analysis,
            "fix_plan": fix_plan,
            "applied": applied,
            "errors": errors,
            "fingerprint": fingerprint,
        })

        if not apply_success:
            logger.error("Applying fixes failed: %s", errors)
            _maybe_escalate(on_escalation, error_analysis, fix_history,
                            reason="apply_failed")
            return False, last_stdout, last_stderr, fix_history

    logger.warning("Exceeded max_retries=%d", max_retries)
    _maybe_escalate(on_escalation, fix_history[-1]["error_analysis"] if fix_history else {},
                    fix_history, reason="max_retries")
    return False, last_stdout, last_stderr, fix_history


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _dispatch_fix(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    """Route a single fix to the correct handler. Returns (ok, message)."""
    if fix.type == FixType.INSTALL_DEPENDENCY.value:
        return _apply_install_dependency(fix, dry_run)
    if fix.type == FixType.MODIFY_FILE.value:
        return _apply_write_file(fix, dry_run, op="modify")
    if fix.type == FixType.CREATE_FILE.value:
        return _apply_write_file(fix, dry_run, op="create")
    if fix.type == FixType.DELETE_FILE.value:
        return _apply_delete_file(fix, dry_run)
    if fix.type == FixType.RUN_COMMAND.value:
        return _apply_run_command(fix, dry_run)
    return False, f"Unknown fix type: {fix.type}"


def _apply_install_dependency(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    pkg = fix.target.strip()
    if not pkg:
        return False, "install_dependency requires a non-empty target"
    if not PIP_NAME_RE.match(pkg):
        return False, f"Unsafe pip package spec rejected: {pkg!r}"
    cmd = f"{_python_exe()} -m pip install {pkg}"
    if dry_run:
        return True, f"[dry-run] would run: {cmd}"
    ok, _, stderr = execute_command(cmd, timeout=120)
    return ok, (f"Installed dependency: {pkg}" if ok
                else f"Failed to install {pkg}: {stderr.strip()[:200]}")


def _apply_write_file(fix: Fix, dry_run: bool, op: str) -> Tuple[bool, str]:
    if not fix.target:
        return False, f"{op}_file requires a target path"
    if fix.content is None or not isinstance(fix.content, str):
        return False, f"{op}_file requires string content"
    if dry_run:
        return True, f"[dry-run] would {op} file: {fix.target} ({len(fix.content)} chars)"
    ok, err, files = create_or_update_file(fix.target, fix.content)
    return ok, (f"{op.title()}d file: {files[0] if files else fix.target}" if ok
                else f"Failed to {op} {fix.target}: {err}")


def _apply_delete_file(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    if not fix.target:
        return False, "delete_file requires a target path"
    if dry_run:
        return True, f"[dry-run] would delete: {fix.target}"
    ok, err = delete_file(fix.target)
    return ok, (f"Deleted file: {fix.target}" if ok
                else f"Failed to delete {fix.target}: {err}")


def _apply_run_command(fix: Fix, dry_run: bool) -> Tuple[bool, str]:
    cmd = fix.action or fix.target
    if not cmd:
        return False, "run_command requires an action or target"
    if _is_dangerous(cmd):
        return False, f"Refused dangerous command: {cmd!r}"
    if dry_run:
        return True, f"[dry-run] would run: {cmd}"
    ok, _, stderr = execute_command(cmd)
    return ok, (f"Executed: {cmd}" if ok
                else f"Command failed: {cmd} -- {stderr.strip()[:200]}")


# ---------------------------------------------------------------------------
# LLM invocation & validation
# ---------------------------------------------------------------------------

from langchain_core.messages import HumanMessage, SystemMessage

def _invoke_llm(error_analysis: Dict, code_context: str, file_path: str) -> Dict:
    """Call the LLM directly with messages — no string templating, no brace bugs."""
    user_text = (
        "Generate fixes for this error:\n\n"
        f"Error Type: {error_analysis.get('error_type', 'Unknown')}\n"
        f"Error Message: {error_analysis.get('error_message', '')}\n"
        f"Root Cause: {error_analysis.get('root_cause', '')}\n"
        f"Severity: {error_analysis.get('severity', 'major')}\n"
        f"Affected Component: {error_analysis.get('affected_component', 'Unknown')}\n"
        f"File Path: {file_path or ''}\n\n"
        f"Code Context:\n{(code_context or '')[:4000]}\n\n"
        f"Suggested Fix Direction: {error_analysis.get('suggested_fix', '')}"
    )

    messages = [
        SystemMessage(content=FIX_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    response = llm.invoke(messages)
    text = getattr(response, "content", "") or ""
    logger.debug("LLM fix raw (300 chars): %s", text[:300])
    try:
        return extract_json_object(text) or {}
    except ValueError:
        return {}
    

    

def _validate_fix_plan(raw: Dict, fallback: FixPlan) -> FixPlan:
    if not isinstance(raw, dict) or "fixes" not in raw:
        return fallback

    raw_fixes = raw.get("fixes") or []
    if not isinstance(raw_fixes, list):
        return fallback

    fixes = [Fix.from_dict(f) for f in raw_fixes[:MAX_FIXES_PER_PLAN]]
    fixes = [f for f in fixes if f is not None]
    if not fixes:
        return fallback

    try:
        confidence = int(raw.get("confidence", fallback.confidence))
    except (TypeError, ValueError):
        confidence = fallback.confidence
    confidence = max(0, min(100, confidence))

    rollback = raw.get("rollback_steps") or fallback.rollback_steps
    if not isinstance(rollback, list):
        rollback = [str(rollback)]

    return FixPlan(
        fixes=fixes,
        verification_command=str(raw.get("verification_command") or fallback.verification_command),
        rollback_steps=[str(s) for s in rollback],
        explanation=str(raw.get("explanation") or fallback.explanation),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fallback fixes (deterministic, safe defaults)
# ---------------------------------------------------------------------------

def _generate_fallback_fixes(error_analysis: Dict, file_path: str = "") -> FixPlan:
    error_type = error_analysis.get("error_type", "UnknownError")
    error_message = error_analysis.get("error_message", "")
    fixes: List[Fix] = []
    verification = ""
    confidence = 30

    if error_type in ("ModuleNotFoundError", "ImportError"):
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        if match:
            module = match.group(1).split(".")[0]  # top-level package
            if PIP_NAME_RE.match(module):
                fixes.append(Fix(
                    type=FixType.INSTALL_DEPENDENCY.value,
                    target=module,
                    action=f"pip install {module}",
                    priority=10,
                ))
                verification = f'{_python_exe()} -c "import {module}"'
                confidence = 75

    elif error_type == "DependencyError":
        # Don't blindly retry; ask LLM next round
        confidence = 15

    elif error_type == "PermissionError":
        # We deliberately DO NOT auto-chmod arbitrary paths.
        confidence = 10

    elif error_type in ("SyntaxError", "IndentationError"):
        # Needs LLM rewrite; fallback can't safely modify the file.
        confidence = 10

    elif error_type == "PortInUseError":
        confidence = 20  # Surfacing to LLM is safer than auto-killing processes

    if not fixes:
        # No safe deterministic action — return empty plan
        return FixPlan(
            fixes=[],
            verification_command=verification,
            rollback_steps=["No deterministic fallback available"],
            explanation=f"No safe automatic fallback for {error_type}",
            confidence=confidence,
        )

    return FixPlan(
        fixes=fixes,
        verification_command=verification,
        rollback_steps=["Uninstall newly added dependencies if needed"],
        explanation=f"Fallback fix for {error_type}",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_exe() -> str:
    """Return the current Python executable, quoted if needed."""
    exe = sys.executable or "python"
    return f'"{exe}"' if " " in exe else exe


def _is_dangerous(command: str) -> bool:
    lowered = command.lower()
    return any(tok in lowered for tok in DANGEROUS_TOKENS)


def _fingerprint_error(analysis: Dict) -> str:
    """Stable hash of (error_type, error_message) for loop detection."""
    raw = f"{analysis.get('error_type', '')}|{analysis.get('error_message', '')}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _maybe_escalate(
    callback: Optional[Callable[[Dict], None]],
    error_analysis: Dict,
    history: List[Dict],
    reason: str,
) -> None:
    if callback is None:
        logger.info("Escalation triggered (reason=%s) — no handler registered", reason)
        return
    try:
        callback({
            "reason": reason,
            "error_analysis": error_analysis,
            "fix_history": history,
        })
    except Exception:
        logger.exception("Escalation handler raised")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Fix Generator...\n")

    # Test 1: ModuleNotFoundError
    print("--- Test 1: ModuleNotFoundError ---")
    err1 = {
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'flask'",
        "root_cause": "Flask is not installed",
        "affected_component": "app.py:1",
        "severity": "major",
        "context": "import flask",
        "suggested_fix": "Install Flask",
        "is_recoverable": True,
    }
    plan1 = generate_fixes(err1)
    print(f"  fixes: {len(plan1['fixes'])}, confidence: {plan1['confidence']}%")
    ok, applied, errors = apply_fixes(plan1, dry_run=True)
    print(f"  dry-run ok={ok}, applied={applied}, errors={errors}\n")

    # Test 2: SyntaxError (no safe deterministic fallback)
    print("--- Test 2: SyntaxError ---")
    err2 = {
        "error_type": "SyntaxError",
        "error_message": "invalid syntax",
        "root_cause": "Incorrect Python syntax",
        "affected_component": "main.py:5",
        "severity": "critical",
        "suggested_fix": "Fix syntax",
        "is_recoverable": True,
    }
    plan2 = generate_fixes(err2, code_context="if x = 5\n    pass", file_path="main.py")
    print(f"  fixes: {len(plan2['fixes'])}, confidence: {plan2['confidence']}%")
    print(f"  explanation: {plan2['explanation']}\n")

    # Test 3: Dangerous command rejection
    print("--- Test 3: Dangerous command rejection ---")
    bad_plan = {
        "fixes": [{"type": "run_command", "target": "danger",
                   "action": "sudo rm -rf /", "priority": 10}],
        "verification_command": "", "rollback_steps": [],
        "explanation": "", "confidence": 90,
    }
    ok, applied, errors = apply_fixes(bad_plan, dry_run=False)
    print(f"  ok={ok}, applied={applied}, errors={errors}")