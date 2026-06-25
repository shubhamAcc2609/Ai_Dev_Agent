"""
Code Generator Module

Responsible for:
- Taking a step description with optional task context (requirement, plan,
  files-so-far) so the LLM can produce real content on attempt 1.
- Using an LLM to generate a structured execution plan
- Validating the plan against a strict, operation-aware schema
- Returning a typed, safe execution plan
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import PurePosixPath, PureWindowsPath
from typing import Optional, Literal, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Operation = Literal["create_file", "update_file", "execute_command", "verify"]

ALLOWED_OPERATIONS: set[str] = {
    "create_file",
    "update_file",
    "execute_command",
    "verify",
}

FILE_OPERATIONS: set[str] = {"create_file", "update_file"}
COMMAND_OPERATIONS: set[str] = {"execute_command"}


@dataclass
class ExecutionPlan:
    operation: Operation
    command: Optional[str]
    file_path: Optional[str]
    file_content: Optional[str]
    verification: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
MAX_FILE_CONTENT_BYTES = 200_000      # 200 KB safeguard
MAX_COMMAND_LENGTH = 1000

# Commands the LLM should never produce (file writes via shell)
# Tokens that indicate the LLM is trying to write files via shell instead
# of using create_file. Note: 'echo' and 'printf' alone are fine — they
# only become file-writes when combined with > or >>, which are blocked
# elsewhere by execution_manager.
FORBIDDEN_COMMAND_TOKENS = (
    ">", ">>",          # output redirection → file write
    "<<",               # heredoc → multi-line file write
    " | tee",           # tee → file write
    "tee ",             # tee → file write (start of line)
    "touch ",           # creates empty file
    "cat >",            # cat with redirect → file write
    "cat >>",
    "heredoc",
)

WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Common placeholder strings we should reject in file_content
PLACEHOLDER_CONTENT_PATTERNS = (
    re.compile(r"^\s*(#\s*)?TODO\s*[:.]?\s*(implement|finish|fill|add code)?\s*$",
               re.IGNORECASE),
    re.compile(r"^\s*pass\s*$"),
    re.compile(r"^\s*\.\.\.\s*$"),
    re.compile(r"^\s*(#|//)\s*(placeholder|empty|stub)\s*$", re.IGNORECASE),
)

REQUIRED_FIELDS: List[str] = [
    "operation",
    "command",
    "file_path",
    "file_content",
    "verification",
    "description",
]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CODE_GENERATOR_SYSTEM_PROMPT = """\
You are a Code Generation Agent.

Your job is to generate the NEXT executable step required to complete the task.

Respond with a valid JSON object only. The first character MUST be `{` and the
last character MUST be `}`. No prose, no markdown fences, no <think> tags.

JSON Schema (every field MUST be present; use null when not applicable):
{
  "operation": "create_file | update_file | execute_command | verify",
  "command": null,
  "file_path": null,
  "file_content": null,
  "verification": "how to verify success",
  "description": "brief explanation of this step"
}

FILE PATH RULES:
- Paths MUST be project-relative (e.g. "main.py", "src/app.py").
- NEVER use absolute paths, "..", leading "/", drive letters (C:\\), or backslashes.
- Assume a workspace already exists; only write inside it.

OPERATION RULES:
- create_file / update_file: REQUIRE both `file_path` and `file_content`.
  Return the COMPLETE updated file in `file_content` (never partial diffs,
  never empty, never placeholder content). Set `command` to null.
- execute_command: REQUIRES a non-empty `command` string. Use only for running
  code, installing dependencies, git, or builds. Set `file_path` and
  `file_content` to null.
- verify: a no-op step that only describes a verification action. Set
  `command`, `file_path`, and `file_content` to null.

CONTEXT USAGE:
The human message may include OVERALL REQUIREMENT, FULL PLAN, and
FILES ALREADY CREATED sections. USE THEM:
- Infer real, working file content from the requirement and plan.
- Do not duplicate files that already exist — update them instead.
- Do not write multiple files with the same purpose under different names.

FORBIDDEN:
- Shell-based file writing: echo, cat, tee, touch, printf, >, >>, heredocs.
- Duplicate files like app_new.py, main_v2.py, final_final.py.
- Empty strings; use null for unused fields.
- Markdown fences or explanations outside the JSON object.
- Shell-based file writing: tee, touch, >, >>, heredocs.
- Stdin piping for input: NO `echo X | ./prog`, NO `prog <<EOF...`,
  NO `cat input.txt | prog`. Pipes (|) are blocked in the sandbox.
- Reading from stdin in test commands. If the program needs input,
  rewrite it to use hardcoded values or argv[N] instead.
- Duplicate files like app_new.py, main_v2.py, final_final.py.
- Empty strings; use null for unused fields.
- Markdown fences or explanations outside the JSON object.
- Never use redirection: `>`, `>>`, `<`, `2>&1`
- Never use background process tricks: `&`, `nohup`, `disown`
- Never use stdin piping for input: `echo X | prog`, `prog <<EOF`
- If verification needs more than what an exit code provides,
  describe the verification in the description field instead of
  trying to capture output via shell tricks.

Always populate `operation`, `verification`, and `description` with non-empty
strings. NEVER leave `file_content` as an empty string or placeholder for
create_file or update_file. NEVER leave `command` as an empty string for
execute_command. Use null only when a field is genuinely not needed.

EXAMPLES:

Example A — create_file with real code:
{
  "operation": "create_file",
  "command": null,
  "file_path": "main.py",
  "file_content": "def find_largest(numbers):\\n    if not numbers:\\n        raise ValueError('empty list')\\n    largest = numbers[0]\\n    for n in numbers[1:]:\\n        if n > largest:\\n            largest = n\\n    return largest\\n\\nif __name__ == '__main__':\\n    print(find_largest([3, 1, 9, 4]))\\n",
  "verification": "python main.py prints 9",
  "description": "Create main entry point with find_largest function"
}

Example B — execute_command:
{
  "operation": "execute_command",
  "command": "python main.py",
  "file_path": null,
  "file_content": null,
  "verification": "Exit code 0 with expected output",
  "description": "Run the program to verify it works"
}

Example C — verify (no-op):
{
  "operation": "verify",
  "command": null,
  "file_path": null,
  "file_content": null,
  "verification": "Inspect main.py to confirm find_largest handles empty input",
  "description": "Verify edge-case handling"
}

ANTI-EXAMPLES (DO NOT DO THESE):

X Bad — empty file_content:
{"operation": "create_file", "file_path": "main.py", "file_content": "", ...}

X Bad — placeholder content:
{"operation": "create_file", "file_path": "main.py",
 "file_content": "# TODO: implement", ...}

X Bad — file_content is just "pass":
{"operation": "create_file", "file_path": "main.py",
 "file_content": "pass", ...}

If you do not have enough information to write real content, infer it from
the OVERALL REQUIREMENT and FULL PLAN sections of the user message. Never
emit empty or placeholder content.
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
    Generate a structured execution plan for the given step.

    Args:
        step_description: The current step to execute (from the planner).
        requirement: The overall user requirement (helps the LLM produce
            meaningful file content on attempt 1).
        plan_overview: The full list of plan steps (for context).
        files_so_far: Files already created by previous steps (so the LLM
            doesn't duplicate them).

    Retries on transient LLM/JSON failures up to MAX_RETRIES times, feeding
    the previous error back to the LLM for self-correction.

    Raises:
        ValueError: If the LLM repeatedly fails to produce a valid plan.
    """
    if not isinstance(step_description, str) or not step_description.strip():
        raise ValueError("step_description must be a non-empty string")

    logger.info("Generating execution plan for step: %r", step_description)
    last_validation_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response_text = _invoke_llm(
                step_description=step_description,
                requirement=requirement or "",
                plan_overview=plan_overview or [],
                files_so_far=files_so_far or [],
                attempt=attempt,
                last_error=last_validation_error,
            )
            plan_dict = _extract_json_from_response(response_text)
            plan = _validate_and_build_plan(plan_dict)
            logger.info("Plan generated on attempt %d: %s",
                        attempt, plan.operation)
            return plan.to_dict()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            last_validation_error = exc
            continue

    raise ValueError(
        f"Code Generator failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_validation_error}"
    )


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _invoke_llm(
    step_description: str,
    requirement: str,
    plan_overview: List[str],
    files_so_far: List[str],
    attempt: int,
    last_error: Optional[Exception],
) -> str:
    """Call the LLM with full task context for better first-shot accuracy."""
    context_parts: List[str] = []
    if requirement:
        context_parts.append(f"OVERALL REQUIREMENT:\n{requirement}")
    if plan_overview:
        numbered = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(plan_overview)
        )
        context_parts.append(f"FULL PLAN:\n{numbered}")
    if files_so_far:
        context_parts.append(
            "FILES ALREADY CREATED:\n"
            + "\n".join(f"  - {f}" for f in files_so_far)
        )

    context_block = "\n\n".join(context_parts)

    user_content = (
        (f"{context_block}\n\n" if context_block else "")
        + f"CURRENT STEP:\n{step_description}\n\n"
        "Generate the next executable action only. If this step creates a "
        "file, write the COMPLETE working file content based on the OVERALL "
        "REQUIREMENT — never leave it empty or use placeholders."
    )

    if attempt > 1 and last_error is not None:
        user_content += (
            f"\n\nYour previous response was rejected with this error:\n"
            f"{last_error}\n\n"
            "CORRECTIONS REQUIRED:\n"
            "- Include ALL six fields (operation, command, file_path, "
            "file_content, verification, description).\n"
            "- Use null (not empty strings) for fields that don't apply.\n"
            "- For create_file/update_file: file_content MUST be a complete, "
            "working file — NOT an empty string, NOT 'pass', NOT a TODO, "
            "NOT a placeholder comment.\n"
            "- For execute_command: command MUST be a real shell command.\n"
            "- Refer to the OVERALL REQUIREMENT above to know WHAT to write."
        )

    messages = [
        SystemMessage(content=CODE_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    text = getattr(response, "content", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned empty content")
    logger.debug("LLM raw response (truncated): %s", text[:500])
    return text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _remove_thinking_blocks(text: str) -> str:
    """Strip all <think>...</think> blocks."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_json_from_response(response_text: str) -> dict:
    """Robustly extract a JSON object from arbitrary LLM output."""
    if not response_text:
        raise ValueError("Empty LLM response")

    cleaned = _remove_thinking_blocks(response_text)

    # 1) Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2) Try fenced code block
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Balanced-brace fallback
    obj = _extract_first_balanced_object(cleaned)
    if obj is not None:
        return obj

    raise ValueError(
        f"Failed to parse JSON from LLM response.\n"
        f"--- Response (first 500 chars) ---\n{cleaned[:500]}"
    )


def _extract_first_balanced_object(text: str) -> Optional[dict]:
    """Find the first balanced {...} block, respecting strings/escapes."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_and_build_plan(plan: dict) -> ExecutionPlan:
    """
    Validate raw plan dict and return a typed ExecutionPlan.

    Missing keys are treated as null — we only require the keys the
    specific *operation* actually needs.
    """
    if not isinstance(plan, dict):
        raise ValueError(
            f"Plan must be a JSON object, got {type(plan).__name__}"
        )

    # Fill missing optional fields with None
    normalized = {key: plan.get(key) for key in REQUIRED_FIELDS}

    # Operation must always be present and valid
    operation = normalized["operation"]
    if not operation:
        raise ValueError("Missing required field: operation")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"Invalid operation '{operation}'. "
            f"Allowed: {sorted(ALLOWED_OPERATIONS)}"
        )

    # Verification + description must be non-empty strings
    for text_field in ("verification", "description"):
        value = normalized[text_field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{text_field}' must be a non-empty string")

    # Operation-specific schema enforcement
    if operation in FILE_OPERATIONS:
        _require_file_fields(normalized)
    elif operation in COMMAND_OPERATIONS:
        _require_command_field(normalized)
    elif operation == "verify":
        if (normalized["file_path"]
                or normalized["file_content"]
                or normalized["command"]):
            raise ValueError(
                "'verify' operation must not include file_path, "
                "file_content, or command"
            )

    _validate_file_path(normalized.get("file_path"))
    _validate_file_content(normalized.get("file_content"))
    _validate_command(normalized.get("command"))

    return ExecutionPlan(
        operation=operation,
        command=normalized["command"],
        file_path=normalized["file_path"],
        file_content=normalized["file_content"],
        verification=normalized["verification"].strip(),
        description=normalized["description"].strip(),
    )


def _require_file_fields(plan: dict) -> None:
    if not plan.get("file_path"):
        raise ValueError(
            f"'{plan['operation']}' requires a non-empty 'file_path'"
        )
    content = plan.get("file_content")
    if content is None or (isinstance(content, str) and not content.strip()):
        raise ValueError(
            f"'{plan['operation']}' requires non-empty 'file_content'"
        )

    # Reject placeholder content
    if isinstance(content, str):
        stripped = content.strip()
        for pattern in PLACEHOLDER_CONTENT_PATTERNS:
            if pattern.match(stripped):
                raise ValueError(
                    f"'{plan['operation']}' file_content is a placeholder "
                    f"({stripped[:60]!r}); provide real content"
                )

    if plan.get("command"):
        raise ValueError(
            f"'{plan['operation']}' must not include a 'command'"
        )


def _require_command_field(plan: dict) -> None:
    cmd = plan.get("command")
    if not cmd or not str(cmd).strip():
        raise ValueError("'execute_command' requires a non-empty 'command'")
    if plan.get("file_path") or plan.get("file_content"):
        raise ValueError(
            "'execute_command' must not include file_path or file_content"
        )


def _validate_file_path(file_path: Optional[str]) -> None:
    """Strict security validation for file paths."""
    if not file_path:
        return

    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")

    if "\x00" in file_path:
        raise ValueError("Null byte in file_path")

    if "\\" in file_path:
        raise ValueError(
            f"Backslashes not allowed in file_path: {file_path!r}"
        )

    if file_path.startswith("/") or file_path.startswith("~"):
        raise ValueError(f"Path must be project-relative: {file_path!r}")

    if re.match(r"^[A-Za-z]:", file_path):
        raise ValueError(f"Windows drive letter not allowed: {file_path!r}")

    posix = PurePosixPath(file_path)
    if posix.is_absolute():
        raise ValueError(f"Absolute path not allowed: {file_path!r}")

    if any(part == ".." for part in posix.parts):
        raise ValueError(f"Path traversal detected: {file_path!r}")

    win = PureWindowsPath(file_path)
    if win.is_absolute() or win.drive:
        raise ValueError(f"Windows absolute path detected: {file_path!r}")

    for part in posix.parts:
        stem = part.split(".")[0].lower()
        if stem in WINDOWS_RESERVED:
            raise ValueError(
                f"Reserved Windows filename in path: {file_path!r}"
            )


def _validate_file_content(file_content: Optional[str]) -> None:
    if file_content is None:
        return
    if not isinstance(file_content, str):
        raise ValueError("file_content must be a string or null")
    if len(file_content.encode("utf-8")) > MAX_FILE_CONTENT_BYTES:
        raise ValueError(
            f"file_content exceeds {MAX_FILE_CONTENT_BYTES} bytes"
        )


def _validate_command(command: Optional[str]) -> None:
    """Trust-but-verify: ensure LLM didn't sneak in shell write tricks."""
    if command is None:
        return
    if not isinstance(command, str):
        raise ValueError("command must be a string or null")

    stripped = command.strip()
    if not stripped:
        raise ValueError("command must be non-empty when provided")

    if len(stripped) > MAX_COMMAND_LENGTH:
        raise ValueError(
            f"command exceeds {MAX_COMMAND_LENGTH} characters"
        )

    lowered = stripped.lower()
    for token in FORBIDDEN_COMMAND_TOKENS:
        if token in lowered:
            raise ValueError(
                f"command contains forbidden token {token!r}: {stripped!r}"
            )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Code Generator with context...\n")

    plan = generate_execution_plan(
        step_description="Create main.py with the solution",
        requirement=(
            "A python program to find the largest number in a list "
            "without using sorting"
        ),
        plan_overview=[
            "Create main.py with the solution",
            "Test the program with sample inputs",
        ],
        files_so_far=[],
    )

    print(json.dumps(plan, indent=2))