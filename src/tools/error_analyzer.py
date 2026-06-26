"""
Error Analyzer

Classifies execution failures using regex patterns + LLM analysis.
Returns a structured ErrorAnalysis dict that downstream components
(fix_generator, executor) use to decide what to do next.

Never raises — failures bubble up as low-confidence "UnknownError"
results so the agent loop stays alive.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_LLM_INPUT_CHARS = 6000

SEVERITY_SCORE = {"critical": 3, "major": 2, "minor": 1}

# Errors the agent has a realistic shot at fixing without human help
RECOVERABLE_ERRORS = {
    "ModuleNotFoundError", "ImportError", "DependencyError",
    "FileNotFoundError", "SyntaxError", "IndentationError",
    "NameError", "AttributeError", "TypeError", "ValueError",
    "KeyError", "IndexError", "PortInUseError",
}

# Errors that warrant CRITICAL severity regardless of context
CRITICAL_ERRORS = {"MemoryError", "SegmentationFault", "SignalTermination"}

# Ordered classification patterns — first match wins, most specific first.
# Matches anchored to Python traceback shapes to keep false positives low.
ERROR_PATTERNS = [
    ("ModuleNotFoundError", re.compile(r"\bModuleNotFoundError\b|No module named ['\"]")),
    ("ImportError",         re.compile(r"\bImportError\b|cannot import name")),
    ("IndentationError",    re.compile(r"\bIndentationError\b")),
    ("SyntaxError",         re.compile(r"\bSyntaxError\b|invalid syntax")),
    ("PermissionError",     re.compile(r"Permission denied|Access is denied", re.IGNORECASE)),
    ("FileNotFoundError",   re.compile(r"\bFileNotFoundError\b|No such file or directory")),
    ("TimeoutError",        re.compile(r"\bTimeoutError\b|timed out", re.IGNORECASE)),
    ("ConnectionError",     re.compile(r"\bConnectionError\b|Connection (refused|reset)", re.IGNORECASE)),
    ("PortInUseError",      re.compile(r"Address already in use|EADDRINUSE", re.IGNORECASE)),
    ("DependencyError",     re.compile(r"Could not find a version|No matching distribution", re.IGNORECASE)),
    ("MemoryError",         re.compile(r"\bMemoryError\b|Killed\s*$", re.IGNORECASE | re.MULTILINE)),
    ("SegmentationFault",   re.compile(r"Segmentation fault|SIGSEGV", re.IGNORECASE)),
    ("KeyError",            re.compile(r"\bKeyError\b")),
    ("ValueError",          re.compile(r"\bValueError\b")),
    ("TypeError",           re.compile(r"\bTypeError\b")),
    ("AttributeError",      re.compile(r"\bAttributeError\b")),
    ("NameError",           re.compile(r"\bNameError\b")),
    ("RuntimeError",        re.compile(r"\bRuntimeError\b")),
]

# "ActualError: message" — Python's real error sits at the END of a traceback
FINAL_ERROR_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9_]*(?:Error|Exception):.*)$",
    re.MULTILINE,
)

# Suggested next actions per error type
FIX_HINTS = {
    "ModuleNotFoundError": "Install the missing package with `pip install <module>`.",
    "ImportError":         "Check the import path or install the right package version.",
    "SyntaxError":         "Fix the syntax error at the indicated line.",
    "IndentationError":    "Fix indentation; use consistent spaces or tabs.",
    "FileNotFoundError":   "Verify the file path is correct and relative to the workspace.",
    "PermissionError":     "Check file/directory permissions.",
    "TimeoutError":        "Increase the timeout or optimize the slow operation.",
    "ConnectionError":     "Verify the target host/port is reachable.",
    "PortInUseError":      "Free the port or use a different one.",
    "DependencyError":     "Pin a compatible version in requirements.txt.",
    "MemoryError":         "Reduce memory usage or run on a host with more RAM.",
}


@dataclass
class ErrorAnalysis:
    error_type: str
    error_message: str
    root_cause: str
    affected_component: str
    severity: str
    context: str
    suggested_fix: str
    is_recoverable: bool


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert error analysis system for an autonomous Software Development Agent.

Examine the execution failure and respond with ONLY a single valid JSON object —
no markdown, no prose, no <think> tags.

Required keys:
- "error_type": Specific category (e.g. "ModuleNotFoundError", "SyntaxError").
- "error_message": The actual error line, one line, concise.
- "root_cause": Short analysis of WHY this happened.
- "affected_component": File / module / line (e.g. "main.py:5").
- "severity": Exactly one of "critical", "major", "minor".
- "context": Up to 10 relevant traceback lines.
- "suggested_fix": Concrete next action.
- "is_recoverable": true if an agent can fix this without human input.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_execution_failure(
    stdout: str,
    stderr: str,
    command: str = "",
    return_code: Optional[int] = None,
) -> Dict:
    """
    Analyze a failed command and return a structured ErrorAnalysis as dict.

    Never raises — LLM failures fall back to the regex-based analysis so
    the agent's retry loop keeps moving.
    """
    stdout, stderr, command = stdout or "", stderr or "", command or ""

    logger.info(
        "Analyzing failure (cmd=%r, rc=%s, stdout=%dB, stderr=%dB)",
        command, return_code, len(stdout), len(stderr),
    )

    # No output to work with
    if not stdout.strip() and not stderr.strip():
        return _empty_output_result(return_code)

    # Always build the deterministic fallback first — used if LLM is unavailable
    fallback = _build_fallback(stdout, stderr, command, return_code)

    llm_result = _try_llm_analysis(command, stdout, stderr)
    if llm_result is None:
        return _to_dict(fallback)

    merged = _merge_with_fallback(llm_result, fallback)
    logger.info(
        "Analysis complete: type=%s severity=%s recoverable=%s",
        merged.error_type, merged.severity, merged.is_recoverable,
    )
    return _to_dict(merged)


def classify_error_type(output: str) -> str:
    """Match output against known error patterns. Returns 'UnclassifiedError' if no match."""
    if not output:
        return "UnclassifiedError"
    for name, pattern in ERROR_PATTERNS:
        if pattern.search(output):
            return name
    return "UnclassifiedError"


def extract_error_message(stdout: str, stderr: str) -> str:
    """
    Find the actual error message line.

    Python tracebacks end with the real error — "Traceback (most recent..." is
    just the header. We search backwards for "SomethingError: details" pattern.
    """
    for stream in (stderr, stdout):
        if not stream:
            continue
        matches = FINAL_ERROR_RE.findall(stream)
        if matches:
            return matches[-1].strip()

    # No structured error line — return the last non-empty line we can find
    for stream in (stderr, stdout):
        for line in reversed(stream.splitlines()):
            if line.strip():
                return line.strip()

    return "No error message captured"


def extract_error_context(output: str, window: int = 3) -> str:
    """Return up to `window*2+1` lines centered on the actual error line."""
    if not output:
        return ""

    lines = output.splitlines()
    # Walk backwards to find the last "SomethingError:" line — that's the anchor
    anchor = -1
    for i in range(len(lines) - 1, -1, -1):
        if FINAL_ERROR_RE.match(lines[i]):
            anchor = i
            break

    if anchor == -1:
        # No anchor found — return the tail, which usually contains the error
        return "\n".join(lines[-(window * 2 + 1):])

    start = max(0, anchor - window)
    end = min(len(lines), anchor + window + 1)
    return "\n".join(lines[start:end])


def get_severity_score(analysis: Dict) -> int:
    """Numeric severity for sorting. Returns 2 (major) for unknown values."""
    return SEVERITY_SCORE.get(str(analysis.get("severity", "")).lower(), 2)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _to_dict(analysis: ErrorAnalysis) -> Dict:
    return {
        "error_type": analysis.error_type,
        "error_message": analysis.error_message,
        "root_cause": analysis.root_cause,
        "affected_component": analysis.affected_component,
        "severity": analysis.severity,
        "context": analysis.context,
        "suggested_fix": analysis.suggested_fix,
        "is_recoverable": analysis.is_recoverable,
    }


def _empty_output_result(return_code: Optional[int]) -> Dict:
    """Build a result for the 'process produced no output' case."""
    if return_code is not None and return_code < 0:
        # POSIX: negative rc = killed by signal
        return _to_dict(ErrorAnalysis(
            error_type="SignalTermination",
            error_message=f"Process killed by signal {-return_code}",
            root_cause="OS terminated the process (possible OOM or external kill)",
            affected_component="Unknown",
            severity="critical",
            context="",
            suggested_fix="Check system logs and memory usage",
            is_recoverable=False,
        ))

    return _to_dict(ErrorAnalysis(
        error_type="UnknownError",
        error_message="No output captured",
        root_cause="Command failed without producing diagnostic output",
        affected_component="Unknown",
        severity="major",
        context="",
        suggested_fix="Re-run with verbose flags or debug logging enabled",
        is_recoverable=False,
    ))


def _build_fallback(
    stdout: str, stderr: str, command: str, return_code: Optional[int],
) -> ErrorAnalysis:
    """Regex-based analysis used as the deterministic fallback."""
    combined = f"{stdout}\n{stderr}"
    error_type = classify_error_type(combined)
    error_message = extract_error_message(stdout, stderr)
    context = extract_error_context(combined, window=4)

    file_line = re.search(r'File "([^"]+)", line (\d+)', combined)
    affected = f"{file_line.group(1)}:{file_line.group(2)}" if file_line else "Unknown"

    severity = "critical" if (
        error_type in CRITICAL_ERRORS or (return_code is not None and return_code < 0)
    ) else ("minor" if error_type == "AssertionError" else "major")

    base_hint = FIX_HINTS.get(error_type, f"Investigate {error_type}: {error_message}")
    suggested = f"{base_hint} (failed command: `{command}`)" if command else base_hint

    return ErrorAnalysis(
        error_type=error_type,
        error_message=error_message,
        root_cause=f"Pattern-based analysis detected {error_type}",
        affected_component=affected,
        severity=severity,
        context=context,
        suggested_fix=suggested,
        is_recoverable=error_type in RECOVERABLE_ERRORS,
    )


def _try_llm_analysis(command: str, stdout: str, stderr: str) -> Optional[Dict]:
    """
    Call the LLM for a deeper analysis. Returns None on any failure
    (network, parsing, etc.) so the caller can fall back gracefully.
    """
    combined = _build_combined_output(stdout, stderr)
    user_text = (
        f"Analyze this execution failure:\n\n"
        f"Command: {command}\n\n"
        f"Output:\n{combined}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ])
    except Exception as exc:
        logger.warning("LLM analysis call failed: %s", exc)
        return None

    text = getattr(response, "content", "") or ""
    try:
        parsed = extract_json_object(text)
    except ValueError as exc:
        logger.warning("Failed to parse LLM analysis: %s", exc)
        return None

    return parsed if isinstance(parsed, dict) else None


def _build_combined_output(stdout: str, stderr: str) -> str:
    """Combine streams with tail-preserving truncation to fit token budget."""
    half = MAX_LLM_INPUT_CHARS // 2
    return (
        f"STDOUT:\n{_tail(stdout, half)}\n\n"
        f"STDERR:\n{_tail(stderr, half)}"
    )


def _tail(text: str, budget: int) -> str:
    """Keep the last `budget` characters; mark if we cut anything."""
    if len(text) <= budget:
        return text
    return "...[truncated]...\n" + text[-budget:]


def _merge_with_fallback(
    llm_result: Dict, fallback: ErrorAnalysis,
) -> ErrorAnalysis:
    """
    Build the final analysis by preferring LLM values when valid,
    falling back to the regex-based analysis otherwise.
    """
    def _pick_str(key: str, default: str) -> str:
        val = llm_result.get(key)
        return val.strip() if isinstance(val, str) and val.strip() else default

    severity = str(llm_result.get("severity", "")).lower().strip()
    if severity not in SEVERITY_SCORE:
        severity = fallback.severity

    return ErrorAnalysis(
        error_type=_pick_str("error_type", fallback.error_type),
        error_message=_pick_str("error_message", fallback.error_message),
        root_cause=_pick_str("root_cause", fallback.root_cause),
        affected_component=_pick_str("affected_component", fallback.affected_component),
        severity=severity,
        context=_pick_str("context", fallback.context),
        suggested_fix=_pick_str("suggested_fix", fallback.suggested_fix),
        is_recoverable=_coerce_bool(llm_result.get("is_recoverable"), fallback.is_recoverable),
    )


def _coerce_bool(value, default: bool) -> bool:
    """Convert common truthy/falsy representations; fall back on uncertainty."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default

