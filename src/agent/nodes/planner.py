"""
Planner Node

Generates an execution plan from the user's requirement, along with
classification metadata the Router uses to dispatch to the right executor.

The metadata fields (project_type, language, needs_compilation, needs_server,
needs_dependencies) let downstream nodes decide what to do without their
own LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.config import llm
from src.agent.nodes.models import ArchitecturePlan
from src.agent.state import AgentState
from src.tools.library_manager import get_planner_context
from src.utils.json_parser import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PLAN_STEPS = 15
MIN_PLAN_STEPS = 1

VALID_PROJECT_TYPES = {"script", "utility", "application", "compiled", "unknown"}
COMPILED_LANGUAGES = {"c", "cpp", "rust", "go", "java", "csharp", "swift", "kotlin"}

# Language aliases — LLMs sometimes use the casual name
LANGUAGE_ALIASES = {
    "c++": "cpp", "cxx": "cpp",
    "c#": "csharp", "cs": "csharp",
    "js": "javascript", "ts": "typescript",
    "py": "python",
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Lead System Architect for an autonomous Software Development Agent.

Break the user's requirement into a STRICTLY ORDERED list of atomic, executable
steps AND classify the project so downstream agents can specialize.

COMPLEXITY SCALING (match plan size to problem size):

A. SIMPLE SCRIPT / ALGORITHM (project_type: "script")
   Examples: "find largest number", "FizzBuzz", "check palindrome".
   → 2-4 steps, one file, stdlib only, no requirements.txt.

B. UTILITY / CLI TOOL (project_type: "utility")
   Examples: "CSV-to-JSON converter", "log analyzer", "password generator".
   → 3-6 steps, 1-2 files, stdlib (or a single trusted package).

C. APPLICATION / API / WEB SERVICE (project_type: "application")
   Examples: "REST API for users", "Flask blog", "FastAPI weather endpoint".
   → 5-10 steps, requirements.txt + install + run/serve + endpoint verification.

D. COMPILED LANGUAGE PROGRAM (project_type: "compiled")
   Examples: "C program to print primes", "Rust hash map demo".
   → 2-4 steps, one source file, compile + run.

   

NEVER inflate a Type A problem into Type C. If the requirement fits in one
file with stdlib only, the plan is SMALL.


A project should be classified as project_type='web' if it involves:
- FastAPI, Flask, Django, Streamlit (Python)
- Express, NestJS, Fastify, Koa (Node.js)
- Next.js, Vite, React dev servers
- Any code that starts an HTTP server

For Node.js projects, set:
- language: 'javascript' or 'typescript'
- needs_server: true
- needs_dependencies: true (for npm install)

HARD RULES:
1. Each step is ATOMIC — a single create_file, update_file, or execute_command.
2. Each file is created exactly ONCE.
3. Don't write source code in the plan, just describe the task.
4. Don't add unnecessary steps (no README, venv setup, deploy steps).
5. Include exactly ONE verification step at the end. For web apps, that step
   must probe an HTTP endpoint, not just start the server.
6. File paths are project-relative ("main.py", "src/utils.py"). No absolute
   paths, no "..", no drive letters.
7. Use `python -m uvicorn` to launch FastAPI/Starlette apps.
8. HTML/CSS/static files are project_type "script". Verification is just
   confirming the files exist — don't run `python -m http.server`.

OUTPUT (JSON only, no markdown fences):
{
  "project_type": "script | utility | application | compiled",
  "language": "python | cpp | rust | javascript | java | go | ...",
  "needs_compilation": true | false,
  "needs_server": true | false,
  "needs_dependencies": true | false,
  "steps": ["Step 1", "Step 2", "..."],
  "thought_process": "one-sentence rationale"
}

EXAMPLES:

Simple script:
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
  "thought_process": "Single-file stdlib problem; 2 steps is enough."
}

Web application:
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
  "thought_process": "Web API needs framework, deps, server, and live endpoint probe."
}
"""


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """Generate (or revise) a plan and emit classification metadata."""
    logger.info("--- PLANNER NODE EXECUTING ---")

    requirement = (state.get("requirement") or "").strip()
    if not requirement:
        return _failed_plan("Planner Node: FAILED — no requirement provided")

    feedback = (state.get("plan_feedback") or "").strip()
    existing_plan: List[str] = list(state.get("plan", []) or [])
    user_message = _build_user_message(requirement, feedback, existing_plan)

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        text = getattr(response, "content", "") or ""
        if not text.strip():
            raise ValueError("LLM returned empty content")

        plan_obj = _extract_plan_json(text)
        plan_data = _normalize_plan(plan_obj)
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
# User message
# ---------------------------------------------------------------------------

def _build_user_message(
    requirement: str,
    feedback: str,
    existing_plan: List[str],
) -> str:
    """Build the user message with toolchain context and any replanning feedback."""
    parts = [
        get_planner_context(),
        f"REQUIREMENT TO ANALYZE:\n{requirement}",
    ]

    if feedback:
        prev_plan = "\n".join(
            f"  {i + 1}. {step}" for i, step in enumerate(existing_plan)
        ) or "  (none)"
        parts.append(
            "REPLANNING CONTEXT:\n"
            "The previous plan failed. Produce a corrected plan that addresses "
            "the feedback below. Keep what worked; change only what's needed. "
            "Respect the HOST TOOLCHAIN STATUS above — do not retry with the "
            "same missing tool.\n\n"
            f"Previous plan:\n{prev_plan}\n\n"
            f"Feedback / error:\n{feedback}"
        )

    parts.append(
        "Produce the JSON plan now. Respect the complexity-scaling rules — "
        "do not over-engineer simple problems. Do not use tools marked NOT "
        "INSTALLED. For web apps, the final step must launch the server AND "
        "verify a real HTTP response."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_plan_json(text: str) -> dict:
    """Pull a JSON object out of the LLM response, handling fences and think tags."""
    cleaned = _THINK_RE.sub("", text).strip()

    try:
        return extract_json_object(cleaned)
    except Exception:
        pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(cleaned)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Planner returned unparseable JSON. First 300 chars: {cleaned[:300]!r}")


def _normalize_plan(plan_obj: dict) -> Dict[str, Any]:
    """Validate the plan and return a normalized dict with all fields filled."""
    if not isinstance(plan_obj, dict):
        raise ValueError(f"Plan must be an object, got {type(plan_obj).__name__}")

    # Validate steps via the existing schema
    plan_obj.setdefault("thought_process", "No rationale provided.")
    try:
        validated = ArchitecturePlan.model_validate(plan_obj)
    except Exception as exc:
        raise ValueError(f"Plan failed schema validation: {exc}") from exc

    steps = _dedupe_steps(validated.steps)
    if len(steps) < MIN_PLAN_STEPS:
        raise ValueError("Plan has no valid steps after cleaning")
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Plan has {len(steps)} steps (max {MAX_PLAN_STEPS}). "
            "Likely over-engineered."
        )

    # Project type
    project_type = str(plan_obj.get("project_type", "unknown")).strip().lower()
    if project_type not in VALID_PROJECT_TYPES:
        project_type = "unknown"

    # Language — normalize via alias table
    language = str(plan_obj.get("language", "unknown")).strip().lower()
    language = LANGUAGE_ALIASES.get(language, language)

    # Boolean flags with inferred fallbacks
    needs_compilation = _to_bool(
        plan_obj.get("needs_compilation"),
        fallback=language in COMPILED_LANGUAGES,
    )
    needs_server = _to_bool(
        plan_obj.get("needs_server"),
        fallback=project_type == "application",
    )
    needs_dependencies = _to_bool(
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
        "rationale": (validated.thought_process or "").strip(),
    }


def _dedupe_steps(steps: List[Any]) -> List[str]:
    """Strip, drop empties, dedupe case-insensitively, preserve order."""
    seen = set()
    cleaned = []
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


def _to_bool(value: Any, fallback: bool) -> bool:
    """Coerce LLM output to bool. Accepts True/False, 'true'/'false', 'yes'/'no', 1/0."""
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


# ---------------------------------------------------------------------------
# Failure helper
# ---------------------------------------------------------------------------

def _failed_plan(message: str) -> dict:
    """Build a state delta representing a planner failure."""
    logger.error(message)
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


