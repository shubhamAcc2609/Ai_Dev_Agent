"""
Error Analyzer Module

Responsible for:
- Examining execution failures from stdout/stderr
- Identifying root causes and error patterns
- Classifying error types via regex + LLM
- Extracting relevant traceback context for fix generation
- Returning a strictly-validated analysis schema
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from src.agent.config import llm
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


SEVERITY_SCORE: Dict[str, int] = {
    Severity.CRITICAL.value: 3,
    Severity.MAJOR.value: 2,
    Severity.MINOR.value: 1,
}

# Truncate logs sent to the LLM to keep tokens bounded
MAX_LLM_INPUT_CHARS = 6000

# Error types we believe an autonomous agent can usually self-heal
RECOVERABLE_ERRORS = {
    "ModuleNotFoundError",
    "ImportError",
    "DependencyError",
    "FileNotFoundError",
    "SyntaxError",
    "IndentationError",
    "NameError",
    "AttributeError",
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "PortInUseError",
}

# Ordered list — first match wins. More specific patterns come first.
# Patterns are anchored to typical traceback shapes to reduce false positives.
ERROR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ModuleNotFoundError", re.compile(r"\bModuleNotFoundError\b|No module named ['\"]")),
    ("ImportError",         re.compile(r"\bImportError\b|cannot import name")),
    ("IndentationError",    re.compile(r"\bIndentationError\b")),
    ("SyntaxError",         re.compile(r"\bSyntaxError\b|invalid syntax")),
    ("PermissionError",     re.compile(r"\bPermissionError\b|Permission denied|Access is denied", re.IGNORECASE)),
    ("FileNotFoundError",   re.compile(r"\bFileNotFoundError\b|No such file or directory")),
    ("TimeoutError",        re.compile(r"\bTimeoutError\b|timed out", re.IGNORECASE)),
    ("ConnectionError",     re.compile(r"\bConnectionError\b|Connection (refused|reset)|Failed to (connect|establish)", re.IGNORECASE)),
    ("PortInUseError",      re.compile(r"Address already in use|EADDRINUSE|bind: address", re.IGNORECASE)),
    ("DependencyError",     re.compile(r"Could not find a version|No matching distribution|requirement.*not satisfied", re.IGNORECASE)),
    ("MemoryError",         re.compile(r"\bMemoryError\b|Killed\s*$|OOM", re.IGNORECASE | re.MULTILINE)),
    ("SegmentationFault",   re.compile(r"Segmentation fault|SIGSEGV", re.IGNORECASE)),
    ("KeyError",            re.compile(r"\bKeyError\b")),
    ("IndexError",          re.compile(r"\bIndexError\b")),
    ("TypeError",           re.compile(r"\bTypeError\b")),
    ("AttributeError",      re.compile(r"\bAttributeError\b")),
    ("NameError",           re.compile(r"\bNameError\b")),
    ("ZeroDivisionError",   re.compile(r"\bZeroDivisionError\b")),
    ("ValueError",          re.compile(r"\bValueError\b")),
    ("RuntimeError",        re.compile(r"\bRuntimeError\b")),
    ("AssertionError",      re.compile(r"\bAssertionError\b")),
    ("OSError",             re.compile(r"\bOSError\b")),
]

REQUIRED_ANALYSIS_KEYS = (
    "error_type", "error_message", "root_cause", "affected_component",
    "severity", "context", "suggested_fix", "is_recoverable",
)


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

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ERROR_ANALYZER_SYSTEM_PROMPT = """\
You are an expert error analysis system for an autonomous Software Development Agent.
Your role is to examine execution failures and identify root causes.

Respond with ONLY a single valid JSON object — no markdown, no prose, no <think> tags.

Required keys (all must be present):
- "error_type": Specific category (e.g. "ModuleNotFoundError", "SyntaxError",
  "PermissionError", "TimeoutError", "DependencyError").
- "error_message": The actual error line from the output (one line, concise).
- "root_cause": Short analysis of WHY this error occurred.
- "affected_component": File / module / line that failed (e.g. "main.py:5").
- "severity": Exactly one of: "critical", "major", "minor".
- "context": Up to 10 relevant lines from the traceback.
- "suggested_fix": Concrete next action (e.g. "Run `pip install fastapi`").
- "is_recoverable": true if an autonomous agent can fix this without human input,
  otherwise false.
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
    Analyze a failed command's output.

    Args:
        stdout: Captured standard output.
        stderr: Captured standard error.
        command: The command string that was executed (for context).
        return_code: Optional process exit code (helps classify timeouts/OOM).

    Returns:
        A dict matching the ErrorAnalysis schema. Always returns a result —
        never raises — to keep the agent loop alive.
    """
    stdout = stdout or ""
    stderr = stderr or ""
    command = command or ""

    logger.info(
        "Analyzing failure (cmd=%r, rc=%s, stdout=%dB, stderr=%dB)",
        command, return_code, len(stdout), len(stderr),
    )

    # No output at all → nothing useful to analyze
    if not stdout.strip() and not stderr.strip():
        return _no_output_analysis(return_code).to_dict()

    combined = _build_combined_output(stdout, stderr)

    # Always compute a deterministic fallback first — used if LLM misbehaves.
    fallback = _fallback_error_analysis(stdout, stderr, command, return_code)

    try:
        llm_result = _invoke_llm(command, combined)
    except Exception as exc:  # network/LLM errors should not crash the agent
        logger.warning("LLM analysis failed (%s); using fallback", exc)
        return fallback.to_dict()

    validated = _validate_llm_analysis(llm_result, fallback)
    logger.info(
        "Analysis complete: type=%s severity=%s recoverable=%s",
        validated.error_type, validated.severity, validated.is_recoverable,
    )
    return validated.to_dict()


def classify_error_type(output: str) -> str:
    """Classify an error from output via ordered regex patterns."""
    if not output:
        return "UnclassifiedError"
    for name, pattern in ERROR_PATTERNS:
        if pattern.search(output):
            return name
    return "UnclassifiedError"


def extract_error_context(output: str, window_size: int = 3) -> str:
    """
    Extract the most relevant slice of output around the actual error.

    Strategy:
    1. Prefer the LAST line matching a final-error pattern (Python's real error
       sits at the bottom of the traceback).
    2. Otherwise, fall back to the last `window_size * 2 + 1` lines.
    """
    if not output:
        return ""

    lines = output.splitlines()
    error_line_re = re.compile(
        r"(error|exception|traceback|failed|fatal)", re.IGNORECASE
    )
    # Lines like 'SomethingError: message' are the true error in Python
    final_error_re = re.compile(r"^\s*[A-Z][A-Za-z0-9_]*(Error|Exception):")

    final_idx = -1
    last_indicator_idx = -1
    for i, line in enumerate(lines):
        if final_error_re.search(line):
            final_idx = i
        elif error_line_re.search(line):
            last_indicator_idx = i

    anchor = final_idx if final_idx != -1 else last_indicator_idx
    if anchor == -1:
        # No marker — return the tail of the output, which is usually most useful
        tail = max(0, len(lines) - (window_size * 2 + 1))
        return "\n".join(lines[tail:])

    start = max(0, anchor - window_size)
    end = min(len(lines), anchor + window_size + 1)
    return "\n".join(lines[start:end])


def extract_error_message(stderr: str, stdout: str) -> str:
    """Pull the most informative single line from the captured streams."""
    final_error_re = re.compile(r"^\s*([A-Z][A-Za-z0-9_]*(?:Error|Exception):.*)$")

    for stream in (stderr, stdout):
        if not stream:
            continue
        # Last line matching SomethingError: ... wins
        match = None
        for line in stream.splitlines():
            m = final_error_re.match(line)
            if m:
                match = m.group(1).strip()
        if match:
            return match

    # Fallback: last non-empty line of stderr, then stdout
    for stream in (stderr, stdout):
        for line in reversed(stream.splitlines()):
            if line.strip():
                return line.strip()

    return "No error message captured"


def get_error_severity_score(analysis: Dict) -> int:
    """Numeric severity score for sorting. Defaults to MAJOR (2)."""
    severity = str(analysis.get("severity", "")).lower()
    return SEVERITY_SCORE.get(severity, SEVERITY_SCORE[Severity.MAJOR.value])


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_combined_output(stdout: str, stderr: str) -> str:
    """Combine streams with truncation to keep LLM tokens bounded."""
    def _tail(text: str, budget: int) -> str:
        if len(text) <= budget:
            return text
        return "...[truncated]...\n" + text[-budget:]

    half = MAX_LLM_INPUT_CHARS // 2
    return f"STDOUT:\n{_tail(stdout, half)}\n\nSTDERR:\n{_tail(stderr, half)}"


from langchain_core.messages import HumanMessage, SystemMessage

def _invoke_llm(command: str, combined_output: str) -> Dict:
    """Call the LLM directly with messages — bypasses str.format brace issues."""
    user_text = (
        f"Analyze this execution failure:\n\n"
        f"Command: {command}\n\n"
        f"Output:\n{combined_output}"
    )
    messages = [
        SystemMessage(content=ERROR_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    response = llm.invoke(messages)
    text = getattr(response, "content", "") or ""
    logger.debug("LLM raw analysis (first 300 chars): %s", text[:300])
    try:
        return extract_json_object(text) or {}
    except ValueError:
        return {}


def _validate_llm_analysis(
    llm_result: Dict, fallback: ErrorAnalysis
) -> ErrorAnalysis:
    """Merge LLM output with fallback, coercing types and filling gaps."""
    if not isinstance(llm_result, dict) or not llm_result:
        return fallback

    merged = fallback.to_dict()
    for key in REQUIRED_ANALYSIS_KEYS:
        if key in llm_result and llm_result[key] not in (None, "", []):
            merged[key] = llm_result[key]

    # Normalize severity
    sev = str(merged.get("severity", "")).lower().strip()
    if sev not in SEVERITY_SCORE:
        sev = fallback.severity
    merged["severity"] = sev

    # Normalize is_recoverable to bool
    merged["is_recoverable"] = _coerce_bool(
        merged.get("is_recoverable"), default=fallback.is_recoverable
    )

    # Ensure all string fields are strings
    for key in ("error_type", "error_message", "root_cause",
                "affected_component", "context", "suggested_fix"):
        if not isinstance(merged.get(key), str):
            merged[key] = str(merged.get(key, ""))

    return ErrorAnalysis(**{k: merged[k] for k in REQUIRED_ANALYSIS_KEYS})


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default


def _no_output_analysis(return_code: Optional[int]) -> ErrorAnalysis:
    if return_code is not None and return_code < 0:
        # Negative rc on POSIX means killed by signal
        return ErrorAnalysis(
            error_type="SignalTermination",
            error_message=f"Process terminated by signal {-return_code}",
            root_cause="Killed by OS (possible OOM, SIGKILL, or external kill)",
            affected_component="Unknown",
            severity=Severity.CRITICAL.value,
            context="",
            suggested_fix="Check system logs / memory usage; re-run with smaller workload",
            is_recoverable=False,
        )
    return ErrorAnalysis(
        error_type="UnknownError",
        error_message="No output captured",
        root_cause="Command failed but produced no diagnostic output",
        affected_component="Unknown",
        severity=Severity.MAJOR.value,
        context="",
        suggested_fix="Re-run with verbose flags or debugging enabled",
        is_recoverable=False,
    )


def _fallback_error_analysis(
    stdout: str, stderr: str, command: str, return_code: Optional[int]
) -> ErrorAnalysis:
    combined = f"{stdout}\n{stderr}"
    error_type = classify_error_type(combined)
    error_msg = extract_error_message(stderr, stdout)
    context = extract_error_context(combined, window_size=4)

    # Heuristic affected component: pull a "File "X", line N" if present
    file_line = re.search(r'File "([^"]+)", line (\d+)', combined)
    affected = f"{file_line.group(1)}:{file_line.group(2)}" if file_line else "Unknown"

    severity = _infer_severity(error_type, return_code)

    return ErrorAnalysis(
        error_type=error_type,
        error_message=error_msg,
        root_cause=f"Pattern-based analysis detected {error_type}",
        affected_component=affected,
        severity=severity,
        context=context,
        suggested_fix=_suggest_fix(error_type, command, error_msg),
        is_recoverable=error_type in RECOVERABLE_ERRORS,
    )


def _infer_severity(error_type: str, return_code: Optional[int]) -> str:
    critical = {"MemoryError", "SegmentationFault", "SignalTermination"}
    minor = {"AssertionError"}
    if error_type in critical:
        return Severity.CRITICAL.value
    if error_type in minor:
        return Severity.MINOR.value
    if return_code is not None and return_code < 0:
        return Severity.CRITICAL.value
    return Severity.MAJOR.value


def _suggest_fix(error_type: str, command: str, error_msg: str) -> str:
    suggestions = {
        "ModuleNotFoundError":
            "Install the missing dependency, e.g. `pip install <module>` "
            "and add it to requirements.txt.",
        "ImportError":
            "Check the import path or install the correct package version.",
        "SyntaxError":
            "Fix the syntax error at the indicated line.",
        "IndentationError":
            "Fix indentation; ensure consistent use of spaces or tabs.",
        "FileNotFoundError":
            "Verify the file path exists relative to the working directory.",
        "PermissionError":
            "Check file/directory permissions or run with appropriate privileges "
            "(but prefer fixing permissions over escalating).",
        "TimeoutError":
            "Increase the timeout or optimize the slow operation.",
        "ConnectionError":
            "Verify the target host/port is reachable; check network/firewall.",
        "PortInUseError":
            "Free the port or configure the app to use a different one.",
        "DependencyError":
            "Pin a compatible version in requirements.txt and reinstall.",
        "MemoryError":
            "Reduce memory usage or run on a host with more RAM.",
        "SegmentationFault":
            "Investigate native dependency; reinstall or downgrade the library.",
    }
    base = suggestions.get(error_type, f"Investigate {error_type}: {error_msg}")
    if command:
        return f"{base} (failed command: `{command}`)"
    return base


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Error Analyzer...\n")

    tests = [
        (
            "ModuleNotFoundError",
            "",
            'Traceback (most recent call last):\n'
            '  File "test.py", line 1, in <module>\n'
            "    import nonexistent_module\n"
            "ModuleNotFoundError: No module named 'nonexistent_module'",
            "python test.py",
        ),
        (
            "SyntaxError",
            "",
            'File "main.py", line 5\n    if x = 5\n       ^\nSyntaxError: invalid syntax',
            "python main.py",
        ),
        (
            "PermissionError",
            "",
            "Permission denied: /root/protected_file.txt",
            "cat /root/protected_file.txt",
        ),
        (
            "DependencyError",
            "",
            "ERROR: Could not find a version that satisfies the requirement numpy==999.0.0\n"
            "ERROR: No matching distribution found for numpy==999.0.0",
            "pip install numpy==999.0.0",
        ),
        (
            "PortInUse",
            "",
            "OSError: [Errno 98] Address already in use",
            "python app.py",
        ),
    ]

    for label, out, err, cmd in tests:
        print(f"--- {label} ---")
        result = analyze_execution_failure(out, err, cmd)
        print(f"  type      : {result['error_type']}")
        print(f"  severity  : {result['severity']}")
        print(f"  message   : {result['error_message']}")
        print(f"  component : {result['affected_component']}")
        print(f"  fix       : {result['suggested_fix']}")
        print(f"  recover?  : {result['is_recoverable']}\n")