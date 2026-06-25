"""
Planner Node

Analyzes the user requirement (and optional replanning feedback) and produces:
- A strictly-ordered, atomic execution plan
- Structured metadata: project_type, language, needs_compilation,
  needs_server, needs_dependencies

The metadata fields downstream nodes (especially the Router) consume so they
can route and execute appropriately — without needing their own LLM calls.

Key properties:
- Self-contained prompt with complexity scaling (script/utility/application)
- Uses SystemMessage + HumanMessage directly (no str.format → no brace bugs)
- Strict JSON output schema, validated and deduplicated
- Hard upper bound on plan length to prevent over-engineering
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm
from src.agent.nodes.models import ArchitecturePlan
from src.agent.state import AgentState
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PLAN_STEPS = 15
MIN_PLAN_STEPS = 1

VALID_PROJECT_TYPES = {"script", "utility", "application", "compiled", "unknown"}

# Known languages — used for normalization
KNOWN_LANGUAGES = {
    "python", "javascript", "typescript", "cpp", "c", "rust", "go",
    "java", "csharp", "ruby", "php", "swift", "kotlin", "scala",
    "haskell", "elixir", "lua", "r", "julia", "dart", "shell",
    "html", "css", "sql", "yaml", "json", "markdown",
    "unknown",
}

# Languages that need compilation
COMPILED_LANGUAGES = {"c", "cpp", "rust", "go", "java", "csharp", "swift", "kotlin"}


# ---------------------------------------------------------------------------
# Inline system prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are the Lead System Architect for an autonomous Software Development Agent.

Your job is to break the user's requirement into a STRICTLY ORDERED list of
atomic, executable steps AND classify the project so downstream agents can
specialize their behavior.

═══════════════════════════════════════════════════════════════════════
COMPLEXITY SCALING (CRITICAL — read carefully)
═══════════════════════════════════════════════════════════════════════

Choose plan size based on problem type:

A. SIMPLE SCRIPT / ALGORITHM
   Examples: "find largest number", "reverse a string", "FizzBuzz",
   "calculate factorial", "check palindrome", "fibonacci sequence".
   → 2-4 steps total
   → ONE file, stdlib only, no frameworks, no requirements.txt
   → project_type: "script"

B. SMALL UTILITY / CLI TOOL
   Examples: "CSV-to-JSON converter", "log analyzer", "password generator".
   → 3-6 steps
   → 1-2 files, stdlib (or a single trusted package)
   → project_type: "utility"

C. APPLICATION / API / WEB SERVICE
   Examples: "REST API for users", "Flask blog", "FastAPI weather endpoint".
   → 5-10 steps
   → Multiple files, real dependencies, framework setup
   → Include requirements.txt + install + run/serve + endpoint verification
   → project_type: "application"

D. COMPILED LANGUAGE PROGRAM
   Examples: "C program to print primes", "Rust hash map demo",
   "C++ binary search tree".
   → 2-4 steps
   → ONE source file, compile + run
   → project_type: "compiled"



NEVER inflate a Type A problem into a Type C plan. If the requirement fits
in one file with stdlib only, the plan is SMALL.

═══════════════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════════════

1. Each step is ATOMIC: a single create_file, update_file, or execute_command
   action. Do not combine multiple actions into one step.

2. Each file is created exactly ONCE. Do NOT plan multiple steps that write
   to the same file.

3. Do NOT write source code in the plan. Describe the task only.

4. Do NOT add unnecessary steps (no README, no virtual environment setup,
   no deploy/dockerize unless explicitly requested).

5. Include exactly ONE verification step at the end. For web applications,
   the verification step MUST probe an endpoint with HTTP, not just start
   the server.

6. File paths must be project-relative ("main.py", "src/utils.py"). Never
   absolute, never use ".." or drive letters.

7. Use `python -m uvicorn ...` to launch FastAPI/Starlette apps.

8.HTML/CSS/STATIC FILES:
   - HTML, CSS, JSON, YAML, Markdown files = project_type "script"
   - needs_server=false, needs_compilation=false, needs_dependencies=false
   - Verification step: just confirm files were created with valid content
   - DO NOT run `python -m http.server` for verification — files existing is enough
   - Example: "Create index.html with student form, create styles.css with form styling"

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only — first char `{`, last char `}`)
═══════════════════════════════════════════════════════════════════════

{
  "project_type": "script | utility | application | compiled",
  "language": "python | cpp | rust | javascript | java | go | ...",
  "needs_compilation": true | false,
  "needs_server": true | false,
  "needs_dependencies": true | false,
  "steps": ["Step 1", "Step 2", "..."],
  "thought_process": "brief one-sentence rationale"
}

FIELD DEFINITIONS:
- project_type: classification category from the four above
- language: the primary programming language being used
- needs_compilation: true ONLY for compiled languages (C, C++, Rust, Go, Java)
- needs_server: true ONLY for long-running HTTP servers (uvicorn, gunicorn, flask run)
- needs_dependencies: true ONLY if external packages must be installed (pip/npm)
- steps: ordered list of step descriptions
- thought_process: one-sentence rationale for your classification

═══════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════

Example 1 — Simple script:
Requirement: "A python program to find the largest number in a list without sorting"
{
  "project_type": "script",
  "language": "python",
  "needs_compilation": false,
  "needs_server": false,
  "needs_dependencies": false,
  "steps": [
    "Create main.py with a find_largest function that iterates through the list once, tracking the maximum, plus a __main__ block demonstrating it on a sample list",
    "Run main.py and verify the printed output matches the expected largest value"
  ],
  "thought_process": "Single-file stdlib problem; minimal 2-step plan."
}

Example 2 — Compiled program:
Requirement: "A C program to print first N fibonacci numbers"
{
  "project_type": "compiled",
  "language": "c",
  "needs_compilation": true,
  "needs_server": false,
  "needs_dependencies": false,
  "steps": [
    "Create main.c with a fib(int n) function that prints the first n Fibonacci numbers using a loop, and a main() that calls fib(10) directly with a hardcoded N",
    "Compile with `gcc main.c -o fib` and run `./fib`, verifying the output starts with 0 1 1 2 3 5 8 13 21 34"
  ],
  "thought_process": "Compiled C program with hardcoded test value to avoid stdin piping in sandbox."
}

Example 3 — Web application:
Requirement: "A FastAPI endpoint that returns the current weather for a city"
{
  "project_type": "application",
  "language": "python",
  "needs_compilation": false,
  "needs_server": true,
  "needs_dependencies": true,
  "steps": [
    "Create requirements.txt listing fastapi, uvicorn, and httpx",
    "Install dependencies with `pip install -r requirements.txt`",
    "Create main.py with a FastAPI app exposing GET /weather?city=<name> that calls a public weather API and returns JSON",
    "Run `python -m uvicorn main:app --port 8000` and verify GET /docs returns HTTP 200"
  ],
  "thought_process": "Web API requires framework, dependencies, server, and live endpoint probe."
}
"""


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """Generate (or revise) a plan and emit classification metadata."""
    logger.info("--- PLANNER NODE EXECUTING ---")

    requirement = (state.get("requirement") or "").strip()
    feedback = (state.get("plan_feedback") or "").strip()
    existing_plan: List[str] = list(state.get("plan", []) or [])

    if not requirement:
        msg = "Planner Node: FAILED — no requirement provided."
        logger.error(msg)
        return _failed_plan(msg)

    user_message = _build_user_message(requirement, feedback, existing_plan)

    try:
        plan_data = _invoke_llm_and_parse(user_message)
    except Exception as exc:
        logger.exception("Planner failed")
        return _failed_plan(f"Planner Node: FAILED — {exc}")

    log_message = (
        f"Planner Node: Generated a {len(plan_data['steps'])}-step "
        f"'{plan_data['project_type']}' plan ({plan_data['language']}). "
        f"compilation={plan_data['needs_compilation']}, "
        f"server={plan_data['needs_server']}, "
        f"deps={plan_data['needs_dependencies']}."
    )
    if plan_data.get("rationale"):
        log_message += f" Rationale: {plan_data['rationale']}"
    logger.info("✓ %s", log_message)

    # ─── Return state delta with all classification metadata ─────────
    return {
        "plan": plan_data["steps"],
        "project_type": plan_data["project_type"],
        "language": plan_data["language"],
        "needs_compilation": plan_data["needs_compilation"],
        "needs_server": plan_data["needs_server"],
        "needs_dependencies": plan_data["needs_dependencies"],
        "current_step": 0,
        "logs": [log_message],
        "is_complete": False,
        "last_error": None,
        "retry_count": 0,
        "plan_feedback": None,
    }


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _build_user_message(
    requirement: str,
    feedback: str,
    existing_plan: List[str],
) -> str:
    """Compose the human message — raw string, no templating."""
    parts = [f"REQUIREMENT TO ANALYZE:\n{requirement}"]

    if feedback:
        prev_plan_block = "\n".join(
            f"  {i + 1}. {step}" for i, step in enumerate(existing_plan)
        ) or "  (none)"
        parts.append(
            "REPLANNING CONTEXT:\n"
            "The previous plan failed. Produce a corrected plan that addresses "
            "the feedback below. Keep what worked; change only what's needed.\n\n"
            f"Previous plan:\n{prev_plan_block}\n\n"
            f"Feedback / error:\n{feedback}"
        )

    parts.append(
        "Produce the JSON plan now. Respect the complexity-scaling rules and "
        "fill in ALL classification fields (project_type, language, "
        "needs_compilation, needs_server, needs_dependencies)."
    )
    return "\n\n".join(parts)


def _invoke_llm_and_parse(user_message: str) -> Dict[str, Any]:
    """Call the LLM and return parsed plan with classification metadata."""
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    text = getattr(response, "content", "") or ""
    if not text.strip():
        raise ValueError("LLM returned empty content")
    logger.debug("Planner raw response (300 chars): %s", text[:300])

    plan_obj = _extract_plan_json(text)
    return _validate_and_normalize_plan(plan_obj)


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _extract_plan_json(text: str) -> dict:
    """Robustly extract a JSON object from arbitrary LLM output."""
    cleaned = _THINK_RE.sub("", text).strip()

    # 1) Try the shared utility (Pydantic-aware)
    try:
        return extract_json_object(cleaned)
    except Exception:
        pass

    # 2) Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3) Fenced code block
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 4) Outermost-brace fallback
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Planner returned unparseable JSON. "
        f"First 300 chars: {cleaned[:300]!r}"
    )


def _validate_and_normalize_plan(plan_obj: dict) -> Dict[str, Any]:
    """
    Validate plan dict and return a normalized version with all fields filled.

    Returns:
        Dict with keys: steps, project_type, language, needs_compilation,
        needs_server, needs_dependencies, rationale.
    """
    if not isinstance(plan_obj, dict):
        raise ValueError(
            f"Plan must be a JSON object, got {type(plan_obj).__name__}"
        )

    # ── Steps validation (via existing ArchitecturePlan model) ──────────
    plan_obj.setdefault("thought_process", "No rationale provided.")
    try:
        validated = ArchitecturePlan.model_validate(plan_obj)
        raw_steps = list(validated.steps)
        rationale = (validated.thought_process or "").strip()
    except Exception as exc:
        raise ValueError(f"Plan failed schema validation: {exc}") from exc

    steps = _clean_and_dedupe_steps(raw_steps)
    if len(steps) < MIN_PLAN_STEPS:
        raise ValueError("Plan has no valid steps after cleaning")
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Plan has {len(steps)} steps (max {MAX_PLAN_STEPS}). "
            "Likely over-engineered — re-run with a tighter requirement."
        )

    # ── Project type normalization ──────────────────────────────────────
    project_type = str(plan_obj.get("project_type", "unknown")).strip().lower()
    if project_type not in VALID_PROJECT_TYPES:
        logger.warning(
            "Unknown project_type %r; defaulting to 'unknown'", project_type
        )
        project_type = "unknown"

    # ── Language normalization ──────────────────────────────────────────
    language = str(plan_obj.get("language", "unknown")).strip().lower()
    # Aliases: c++ → cpp, c# → csharp, js → javascript
    language_aliases = {
        "c++": "cpp", "cxx": "cpp",
        "c#": "csharp", "cs": "csharp",
        "js": "javascript", "ts": "typescript",
        "py": "python",
    }
    language = language_aliases.get(language, language)
    if language not in KNOWN_LANGUAGES:
        logger.warning("Unknown language %r; keeping as-is", language)

    # ── Boolean flags with inference fallbacks ──────────────────────────
    needs_compilation = _infer_boolean(
        plan_obj.get("needs_compilation"),
        fallback=language in COMPILED_LANGUAGES,
    )
    needs_server = _infer_boolean(
        plan_obj.get("needs_server"),
        fallback=project_type == "application",
    )
    needs_dependencies = _infer_boolean(
        plan_obj.get("needs_dependencies"),
        fallback=project_type == "application",
    )

    return {
        "steps": steps,
        "project_type": project_type,
        "language": language,
        "needs_compilation": needs_compilation,
        "needs_server": needs_server,
        "needs_dependencies": needs_dependencies,
        "rationale": rationale,
    }


def _infer_boolean(value: Any, fallback: bool) -> bool:
    """
    Coerce LLM output to a bool. Accepts True/False, 'true'/'false',
    'yes'/'no', 1/0. Falls back to provided default if missing/garbage.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def _clean_and_dedupe_steps(steps: List) -> List[str]:
    """Strip whitespace, drop empties, dedupe while preserving order."""
    seen: set = set()
    cleaned: List[str] = []
    for s in steps:
        if not isinstance(s, str):
            continue
        normalized = s.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            logger.info("Dropping duplicate step: %r", normalized)
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


# ---------------------------------------------------------------------------
# Failure helper
# ---------------------------------------------------------------------------

def _failed_plan(message: str) -> dict:
    """Build a state delta representing a planner failure."""
    return {
        "plan": [],
        "project_type": "unknown",
        "language": "unknown",
        "needs_compilation": False,
        "needs_server": False,
        "needs_dependencies": False,
        "current_step": 0,
        "logs": [message],
        "is_complete": True,
        "last_error": message,
        "retry_count": 0,
        "plan_feedback": None,
        "final_status": "failed",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tests = [
        "A python program to find the largest number in a list without using sorting",
        "A C program to print first 10 fibonacci numbers",
        "A FastAPI endpoint that returns the current weather for a city",
    ]

    for req in tests:
        print(f"\n{'=' * 70}\nRequirement: {req}\n{'=' * 70}")
        mock_state = AgentState(
            requirement=req,
            plan=[],
            files=[],
            logs=[],
            current_step=0,
            is_complete=False,
            last_error=None,
            retry_count=0,
            plan_feedback=None,
            user_feedback=None,
        )
        result = planner_node(mock_state)
        print(f"\nproject_type:     {result.get('project_type')}")
        print(f"language:         {result.get('language')}")
        print(f"needs_compilation: {result.get('needs_compilation')}")
        print(f"needs_server:     {result.get('needs_server')}")
        print(f"needs_dependencies: {result.get('needs_dependencies')}")
        print(f"\nPlan:")
        for i, step in enumerate(result.get("plan", []), 1):
            print(f"  {i}. {step}")