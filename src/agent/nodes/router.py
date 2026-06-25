"""
Router Node

Sits between the Planner and the specialized Executors.

Responsibilities:
- Read the Planner's classification metadata (needs_server, needs_compilation,
  project_type, language) from state
- Pick exactly one execution branch: "simple", "compiled", or "web"
- Emit a clear log line explaining WHY this route was chosen
- NEVER crash — always return a valid route, even on malformed input

Design philosophy:
- The Planner is the smart classifier (has LLM context).
- The Router is pure mechanical logic that respects that classification.
- No LLM call here — sub-millisecond, fully deterministic, easy to test.

Adding a new route:
  1. Add the route constant (ROUTE_DATASCIENCE, etc.)
  2. Add it to VALID_ROUTES
  3. Add a decision branch in _decide_route()
  4. Wire the executor in graph.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route constants — keep in sync with graph.py conditional edges
# ---------------------------------------------------------------------------

ROUTE_SIMPLE = "simple"      # Python scripts, basic CLI tools
ROUTE_COMPILED = "compiled"  # C, C++, Rust, Go, Java
ROUTE_WEB = "web"            # FastAPI, Flask, Django

DEFAULT_ROUTE = ROUTE_SIMPLE  # Fall-back for any unclear case

VALID_ROUTES = {ROUTE_SIMPLE, ROUTE_COMPILED, ROUTE_WEB}


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """
    Inspect the Planner's classification and choose an execution route.

    GUARANTEE: This function never raises. Any exception is caught and
    the default route is returned with a warning log.

    Returns a state delta:
        {
            "route": "simple" | "compiled" | "web",
            "logs": ["Router: ..."]
        }
    """
    logger.info("--- ROUTER NODE EXECUTING ---")

    try:
        route, reason = _decide_route(state)
    except Exception as exc:
        # Defensive: any unexpected error → default route + warning
        logger.exception("Router crashed; using default route")
        return {
            "route": DEFAULT_ROUTE,
            "logs": [
                f"Router: ERROR ({exc}); defaulted to '{DEFAULT_ROUTE}'"
            ],
        }

    # Sanity check — should never trigger, but defense in depth
    if route not in VALID_ROUTES:
        logger.error(
            "Router produced invalid route %r; falling back to %r",
            route, DEFAULT_ROUTE,
        )
        route = DEFAULT_ROUTE
        reason = f"invalid route detected; defaulted to '{DEFAULT_ROUTE}'"

    log_msg = f"Router: route='{route}' — {reason}"
    logger.info(log_msg)

    return {
        "route": route,
        "logs": [log_msg],
    }


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decide_route(state: AgentState) -> Tuple[str, str]:
    """
    Apply the routing rules and return (route, human_readable_reason).

    Priority order (highest to lowest):
      1. needs_server=True → WEB (server-aware execution required)
      2. needs_compilation=True → COMPILED (compile+run required)
      3. project_type hint → matching route
      4. language hint → matching route
      5. Default → SIMPLE
    """
    # ─── Extract metadata defensively ────────────────────────────────────
    metadata = _extract_metadata(state)

    needs_server = metadata["needs_server"]
    needs_compilation = metadata["needs_compilation"]
    project_type = metadata["project_type"]
    language = metadata["language"]
    plan_length = metadata["plan_length"]

    # ─── Guard: empty plan ───────────────────────────────────────────────
    if plan_length == 0:
        return DEFAULT_ROUTE, "empty plan; defaulted to 'simple'"

    # ─── Rule 1: needs_server is the strongest signal ────────────────────
    if needs_server:
        return ROUTE_WEB, f"needs_server=True (project_type='{project_type}')"

    # ─── Rule 2: needs_compilation forces compiled branch ────────────────
    if needs_compilation:
        return ROUTE_COMPILED, (
            f"needs_compilation=True "
            f"(language='{language}', project_type='{project_type}')"
        )

    # ─── Rule 3: explicit project_type mapping ───────────────────────────
    project_type_map = {
        "application": ROUTE_WEB,
        "web":         ROUTE_WEB,
        "compiled":    ROUTE_COMPILED,
        "script":      ROUTE_SIMPLE,
        "utility":     ROUTE_SIMPLE,
    }
    if project_type in project_type_map:
        chosen = project_type_map[project_type]
        return chosen, f"project_type='{project_type}' → '{chosen}'"

    # ─── Rule 4: language hint when classification is missing ────────────
    language_map = {
        "c": ROUTE_COMPILED, "cpp": ROUTE_COMPILED,
        "rust": ROUTE_COMPILED, "go": ROUTE_COMPILED,
        "java": ROUTE_COMPILED, "csharp": ROUTE_COMPILED,
        "swift": ROUTE_COMPILED, "kotlin": ROUTE_COMPILED,
    }
    if language in language_map:
        chosen = language_map[language]
        return chosen, (
            f"language='{language}' implies compiled execution"
        )

    # ─── Rule 5: default fallback ────────────────────────────────────────
    return DEFAULT_ROUTE, (
        f"no decisive signals (project_type='{project_type}', "
        f"language='{language}'); defaulted to '{DEFAULT_ROUTE}'"
    )


# ---------------------------------------------------------------------------
# Defensive metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(state: Any) -> Dict[str, Any]:
    """
    Pull routing-relevant fields from state with type-safe defaults.

    Handles every malformed-input scenario:
      - state is None
      - state is missing fields
      - fields have wrong types (str instead of bool, etc.)
      - state is not a dict at all
    """
    # Treat anything non-dict as empty
    if not isinstance(state, dict):
        logger.warning("Router received non-dict state (%s); using defaults",
                       type(state).__name__)
        state = {}

    return {
        "needs_server":      _to_bool(state.get("needs_server"), default=False),
        "needs_compilation": _to_bool(state.get("needs_compilation"), default=False),
        "project_type":      _to_str(state.get("project_type"), default="unknown"),
        "language":          _to_str(state.get("language"), default="unknown"),
        "plan_length":       _to_plan_length(state.get("plan")),
    }


def _to_bool(value: Any, default: bool) -> bool:
    """Coerce a value to bool, accepting common LLM-output formats."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0", ""):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_str(value: Any, default: str) -> str:
    """Coerce a value to a normalized lowercase string."""
    if not isinstance(value, str):
        return default
    cleaned = value.strip().lower()
    return cleaned if cleaned else default


def _to_plan_length(plan: Any) -> int:
    """Safely compute plan length regardless of type."""
    if isinstance(plan, list):
        return len(plan)
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_cases = [
        # (description, state, expected_route)
        (
            "Simple Python script",
            {
                "plan": ["Create main.py", "Run python main.py"],
                "project_type": "script",
                "language": "python",
                "needs_compilation": False,
                "needs_server": False,
            },
            ROUTE_SIMPLE,
        ),
        (
            "C compiled program",
            {
                "plan": ["Create main.c", "gcc main.c -o fib"],
                "project_type": "compiled",
                "language": "c",
                "needs_compilation": True,
                "needs_server": False,
            },
            ROUTE_COMPILED,
        ),
        (
            "FastAPI web app",
            {
                "plan": ["Create app", "Install deps", "uvicorn main:app"],
                "project_type": "application",
                "language": "python",
                "needs_compilation": False,
                "needs_server": True,
            },
            ROUTE_WEB,
        ),
        (
            "Rust web service (needs_server wins over compiled)",
            {
                "plan": ["Build with cargo", "Run server"],
                "project_type": "application",
                "language": "rust",
                "needs_compilation": True,
                "needs_server": True,
            },
            ROUTE_WEB,
        ),
        (
            "Missing classification — fall back on language",
            {
                "plan": ["Create main.cpp"],
                "language": "cpp",
            },
            ROUTE_COMPILED,
        ),
        (
            "Missing everything — graceful default",
            {
                "plan": ["Do something"],
            },
            ROUTE_SIMPLE,
        ),
        (
            "Empty plan — safe default",
            {
                "plan": [],
                "project_type": "compiled",
            },
            ROUTE_SIMPLE,
        ),
        (
            "Garbage types in state (LLM oddity)",
            {
                "plan": ["Step 1"],
                "project_type": "application",
                "needs_server": "yes",         # string instead of bool
                "needs_compilation": 0,        # int instead of bool
            },
            ROUTE_WEB,
        ),
        (
            "None state (extreme defensive case)",
            None,
            ROUTE_SIMPLE,
        ),
        (
            "Non-dict state",
            "not a dict",
            ROUTE_SIMPLE,
        ),
        (
            "Utility tool",
            {
                "plan": ["Create CLI tool"],
                "project_type": "utility",
                "language": "python",
            },
            ROUTE_SIMPLE,
        ),
        (
            "Java program (compiled by language)",
            {
                "plan": ["Create Main.java"],
                "language": "java",
            },
            ROUTE_COMPILED,
        ),
    ]

    print("Testing Router Node...\n")
    passed = 0
    failed_cases = []

    for description, state, expected in test_cases:
        result = router_node(state)
        actual = result["route"]
        verdict = "✓ PASS" if actual == expected else "✗ FAIL"

        if actual == expected:
            passed += 1
        else:
            failed_cases.append((description, expected, actual))

        print(f"{verdict} | {description}")
        print(f"     expected='{expected}', got='{actual}'")
        print(f"     reason: {result['logs'][0]}\n")

    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    if failed_cases:
        print("\nFailed cases:")
        for desc, exp, got in failed_cases:
            print(f"  - {desc}: expected '{exp}', got '{got}'")
    else:
        print("✓ All tests passed!")