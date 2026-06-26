"""
Code Generator

Generates one execution step at a time using the LLM, then validates the
output against a strict operation-aware schema.

The agent calls this once per step. The context arguments (requirement,
plan_overview, files_so_far) give the LLM enough information to write
working file content on the first attempt instead of placeholders.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
MAX_FILE_CONTENT_BYTES = 200_000
MAX_COMMAND_LENGTH = 1000

ALLOWED_OPERATIONS = {"create_file", "update_file", "execute_command", "verify"}

# Tokens that mean the LLM is trying to write files via the shell instead
# of using create_file. Note: bare `echo`/`printf` are fine — they only
# become file-writes with redirection, which execution_manager blocks too.
FORBIDDEN_SHELL_TOKENS = (
    ">", ">>", "<<",        # redirection and heredocs
    "tee ", " | tee",       # pipe to file
    "touch ",               # empty file creation
    "cat >", "cat >>",      # cat with redirect
)

# Reserved Windows filenames (case-insensitive)
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Catches common placeholder patterns the LLM emits when it doesn't know
# what to write: "pass", "...", "# TODO", "# placeholder", etc.
PLACEHOLDER_RE = re.compile(
    r"^\s*(?:"
    r"(?:#|//)\s*(?:TODO|placeholder|empty|stub|implement|FIXME)"
    r"|pass"
    r"|\.\.\."
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class ExecutionPlan:
    operation: str
    command: Optional[str]
    file_path: Optional[str]
    file_content: Optional[str]
    verification: str
    description: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Code Generation Agent.

Your job is to generate the NEXT executable step required to complete the task.

Respond with a valid JSON object only. First character must be `{`, last must
be `}`. No prose, no markdown fences, no <think> tags.

Schema (every field present; use null when not applicable):
{
  "operation": "create_file | update_file | execute_command | verify",
  "command": null,
  "file_path": null,
  "file_content": null,
  "verification": "how to verify success",
  "description": "brief explanation of this step"
}

OPERATION RULES:
- create_file / update_file: requires file_path and file_content (the COMPLETE
  file body — never partial diffs, never empty, never placeholders). command=null.
- execute_command: requires a non-empty command string. file_path and file_content=null.
- verify: a logical checkpoint. All three of command, file_path, file_content=null.

FILE PATH RULES:
- Project-relative only (e.g. "main.py", "src/app.py").
- No absolute paths, no "..", no leading "/", no backslashes, no drive letters.

CONTEXT USAGE:
The human message may include OVERALL REQUIREMENT, FULL PLAN, and FILES
ALREADY CREATED. Use them to write real, working content. Do not duplicate
existing files — update them.

FORBIDDEN:
- Shell-based file writes: tee, touch, >, >>, heredocs.
- Stdin tricks: `echo X | prog`, `prog <<EOF`, `cat in.txt | prog`.
- Background process tricks: `&`, `nohup`, `disown`.
- Redirection: `>`, `>>`, `<`, `2>&1`.
- Duplicate files like app_new.py, main_v2.py, final_final.py.
- Empty strings (use null) or markdown fences around the JSON.

EXAMPLES:

create_file:
{
  "operation": "create_file",
  "command": null,
  "file_path": "main.py",
  "file_content": "def find_largest(numbers):\\n    if not numbers:\\n        raise ValueError('empty list')\\n    largest = numbers[0]\\n    for n in numbers[1:]:\\n        if n > largest:\\n            largest = n\\n    return largest\\n\\nif __name__ == '__main__':\\n    print(find_largest([3, 1, 9, 4]))\\n",
  "verification": "python main.py prints 9",
  "description": "Create main entry point with find_largest function"
}

execute_command:
{
  "operation": "execute_command",
  "command": "python main.py",
  "file_path": null,
  "file_content": null,
  "verification": "Exit code 0 with expected output",
  "description": "Run the program to verify it works"
}

Bad (don't do this):
{"operation": "create_file", "file_path": "main.py", "file_content": "# TODO"}

If you don't have enough information to write real content, infer it from
OVERALL REQUIREMENT and FULL PLAN. Never emit placeholders.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_execution_plan(
    step_description: str,
    requirement: str = "",
    plan_overview: Optional[List[str]] = None,
    files_so_far: Optional[List[str]] = None,
) -> dict:
    """
    Generate a validated execution plan for one step.

    Retries up to MAX_RETRIES times on parsing/validation failures, feeding
    the previous error back to the LLM for self-correction.

    Raises ValueError if all retries fail.
    """
    if not isinstance(step_description, str) or not step_description.strip():
        raise ValueError("step_description must be a non-empty string")

    logger.info("Generating plan for: %r", step_description)

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _call_llm(
                step_description=step_description,
                requirement=requirement,
                plan_overview=plan_overview or [],
                files_so_far=files_so_far or [],
                attempt=attempt,
                last_error=last_error,
            )
            plan_dict = _extract_json(response)
            plan = _validate_plan(plan_dict)
            logger.info("Plan generated on attempt %d (op=%s)",
                        attempt, plan.operation)
            return _to_dict(plan)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            last_error = exc

    raise ValueError(
        f"Code generator failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _call_llm(
    step_description: str,
    requirement: str,
    plan_overview: List[str],
    files_so_far: List[str],
    attempt: int,
    last_error: Optional[Exception],
) -> str:
    """Build the user message with context and call the LLM."""
    sections = []
    if requirement:
        sections.append(f"OVERALL REQUIREMENT:\n{requirement}")
    if plan_overview:
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_overview))
        sections.append(f"FULL PLAN:\n{numbered}")
    if files_so_far:
        sections.append(
            "FILES ALREADY CREATED:\n"
            + "\n".join(f"  - {f}" for f in files_so_far)
        )

    user_text = "\n\n".join(sections + [
        f"CURRENT STEP:\n{step_description}",
        "Generate the next executable action only. If this step creates a "
        "file, write the COMPLETE working file content based on the OVERALL "
        "REQUIREMENT — never leave it empty or use placeholders.",
    ])

    if attempt > 1 and last_error is not None:
        user_text += (
            f"\n\nPrevious attempt rejected with: {last_error}\n"
            "Return a corrected JSON object. Include all six fields. Use null "
            "(not empty strings) for unused fields. For create_file/update_file, "
            "file_content must be a complete working file — never 'pass' or "
            "'# TODO'."
        )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ])
    text = getattr(response, "content", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned empty content")

    logger.debug("LLM raw response (first 500): %s", text[:500])
    return text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _extract_json(response_text: str) -> dict:
    """
    Pull a JSON object out of the LLM's response. Handles plain JSON,
    markdown-fenced JSON, and stray <think> blocks.
    """
    if not response_text:
        raise ValueError("Empty LLM response")

    cleaned = _THINK_RE.sub("", response_text).strip()

    # Try plain parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fenced code block
    fence = _FENCE_RE.search(cleaned)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response. First 500 chars:\n"
        f"{cleaned[:500]}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_plan(plan: dict) -> ExecutionPlan:
    """
    Validate the LLM output and return a typed ExecutionPlan.

    Operation-specific: each operation has its own required fields. We don't
    require fields that don't apply to the operation.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"Plan must be an object, got {type(plan).__name__}")

    operation = plan.get("operation")
    if not operation:
        raise ValueError("Missing 'operation' field")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Invalid operation: {operation!r}")

    # verification and description must always be non-empty
    for field in ("verification", "description"):
        value = plan.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field}' must be a non-empty string")

    file_path = plan.get("file_path")
    file_content = plan.get("file_content")
    command = plan.get("command")

    # Operation-specific schema enforcement
    if operation in ("create_file", "update_file"):
        if not file_path:
            raise ValueError(f"{operation!r} requires file_path")
        if not isinstance(file_content, str) or not file_content.strip():
            raise ValueError(f"{operation!r} requires non-empty file_content")
        if PLACEHOLDER_RE.match(file_content.strip()):
            raise ValueError(
                f"{operation!r} file_content is a placeholder "
                f"({file_content.strip()[:60]!r}); write real content"
            )
        if command:
            raise ValueError(f"{operation!r} must not include a command")

    elif operation == "execute_command":
        if not isinstance(command, str) or not command.strip():
            raise ValueError("execute_command requires a non-empty command")
        if file_path or file_content:
            raise ValueError(
                "execute_command must not include file_path or file_content"
            )

    elif operation == "verify":
        if file_path or file_content or command:
            raise ValueError(
                "verify must not include file_path, file_content, or command"
            )

    _check_file_path(file_path)
    _check_file_content(file_content)
    _check_command(command)

    return ExecutionPlan(
        operation=operation,
        command=command,
        file_path=file_path,
        file_content=file_content,
        verification=plan["verification"].strip(),
        description=plan["description"].strip(),
    )


def _check_file_path(file_path: Optional[str]) -> None:
    """Reject paths that would escape the workspace or hit reserved names."""
    if not file_path:
        return

    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")
    if "\x00" in file_path:
        raise ValueError("Null byte in file_path")
    if "\\" in file_path:
        raise ValueError(f"Backslashes not allowed: {file_path!r}")
    if file_path.startswith(("/", "~")):
        raise ValueError(f"Path must be project-relative: {file_path!r}")
    if re.match(r"^[A-Za-z]:", file_path):
        raise ValueError(f"Drive letter not allowed: {file_path!r}")

    parts = PurePosixPath(file_path).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Path traversal: {file_path!r}")

    for part in parts:
        stem = part.split(".")[0].lower()
        if stem in WINDOWS_RESERVED:
            raise ValueError(f"Reserved filename in path: {file_path!r}")


def _check_file_content(content: Optional[str]) -> None:
    """Enforce size limit on file content."""
    if content is None:
        return
    if not isinstance(content, str):
        raise ValueError("file_content must be a string or null")
    if len(content.encode("utf-8")) > MAX_FILE_CONTENT_BYTES:
        raise ValueError(f"file_content exceeds {MAX_FILE_CONTENT_BYTES} bytes")


def _check_command(command: Optional[str]) -> None:
    """Reject commands that try to write files via shell tricks."""
    if command is None:
        return
    if not isinstance(command, str):
        raise ValueError("command must be a string or null")

    stripped = command.strip()
    if not stripped:
        raise ValueError("command must be non-empty when provided")
    if len(stripped) > MAX_COMMAND_LENGTH:
        raise ValueError(f"command exceeds {MAX_COMMAND_LENGTH} chars")

    lowered = stripped.lower()
    for token in FORBIDDEN_SHELL_TOKENS:
        if token in lowered:
            raise ValueError(f"command contains forbidden token {token!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(plan: ExecutionPlan) -> dict:
    """ExecutionPlan → dict (for the executor)."""
    return {
        "operation": plan.operation,
        "command": plan.command,
        "file_path": plan.file_path,
        "file_content": plan.file_content,
        "verification": plan.verification,
        "description": plan.description,
    }