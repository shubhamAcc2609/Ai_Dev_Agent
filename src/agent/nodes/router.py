"""
Orchestrator Agent

LLM-powered decision node that sits between the Planner and the specialized
Executors.

Responsibilities:
- Read the user requirement + Planner's classification metadata
- Use an LLM to reason about which execution branch best fits the task
- Emit a clear log line explaining WHY this route was chosen (LLM reasoning)

Design philosophy:
- The Planner extracts structured metadata (project_type, language, flags).
- The Orchestrator REASONS over that metadata + the raw requirement and
  decides the execution strategy.
- Fully agentic: the LLM is the single source of truth for routing.
  If the LLM call fails, the workflow fails fast and surfaces the error
  instead of silently degrading.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.agent.state import AgentState
from src.agent.config import llm

logger = logging.getLogger(__name__)

ROUTE_SIMPLE = "simple"      # Python scripts, basic CLI tools
ROUTE_COMPILED = "compiled"  # C, C++, Rust, Go, Java
ROUTE_WEB = "web"            # FastAPI, Flask, Django

VALID_ROUTES = {ROUTE_SIMPLE, ROUTE_COMPILED, ROUTE_WEB}


# ---------------------------------------------------------------------------
# Structured LLM output schema
# ---------------------------------------------------------------------------

class OrchestratorDecision(BaseModel):
    """Schema the LLM must conform to."""
    executor: str = Field(
        ...,
        description="One of: 'simple', 'compiled', 'web'."
    )
    reason: str = Field(
        ...,
        description="One short sentence explaining the choice."
    )
    overrode_planner: bool = Field(
        default=False,
        description="True if the orchestrator disagreed with the planner."
    )


_SYSTEM_PROMPT = """You are the Orchestrator Agent of an autonomous \
software-development workflow.

Your job: given a user requirement and the Planner's classification, \
choose exactly ONE execution branch.

Available executors:
- "simple"   → Python scripts, CLI tools, data utilities (no compile, no server).
- "compiled" → Languages that must be compiled before running: C, C++, Rust, Go, Java, C#, Swift, Kotlin.
- "web"      → Anything that starts an HTTP server: FastAPI, Flask, Django, Node/Express, static HTML served live.

Decision rules (in order of priority):
1. If the requirement clearly describes an HTTP API, server, or web endpoint → "web".
2. If the requirement names a compiled language (C/C++/Rust/Go/Java/...) → "compiled".
3. Otherwise → "simple".

Trust the Planner's metadata, but OVERRIDE it when the raw requirement \
clearly contradicts it (e.g., planner says "simple" but the user asked \
for a FastAPI endpoint). Set overrode_planner=true when you do.

Respond ONLY as JSON matching this schema:
{{
  "executor": "simple" | "compiled" | "web",
  "reason": "<one short sentence>",
  "overrode_planner": true | false
}}
"""

_USER_PROMPT = """User Requirement:
{user_requirement}

Planner Classification:
- project_type: {project_type}
- language: {language}
- needs_server: {needs_server}
- needs_compilation: {needs_compilation}
- needs_dependencies: {needs_dependencies}

Choose the best executor."""



def orchestrator_node(state: AgentState) -> dict:
    """
    LLM-driven routing decision.

    No fallback: if the LLM call fails or returns an invalid executor,
    the exception propagates so the workflow surfaces the real problem
    instead of silently routing to a default.

    Returns a state delta:
        {
            "route": "simple" | "compiled" | "web",
            "logs":  ["Orchestrator: ..."]
        }
    """
    logger.info("--- ORCHESTRATOR AGENT EXECUTING ---")

    metadata = _extract_metadata(state)

    decision = _llm_decide(metadata)
    route = decision.executor.strip().lower()

    if route not in VALID_ROUTES:
        raise ValueError(
            f"Orchestrator LLM returned invalid executor: {route!r}. "
            f"Expected one of {sorted(VALID_ROUTES)}."
        )

    tag = " [OVERRODE PLANNER]" if decision.overrode_planner else ""
    log_msg = f"Orchestrator: route='{route}'{tag} — {decision.reason}"
    logger.info(log_msg)

    return {"route": route, "logs": [log_msg]}



def _llm_decide(metadata: Dict[str, Any]) -> OrchestratorDecision:
    """Call the LLM with structured output. Raises on any failure."""
    model = llm
    structured_llm = model.with_structured_output(OrchestratorDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("user", _USER_PROMPT),
    ])

    chain = prompt | structured_llm

    decision = chain.invoke({
        "user_requirement":    metadata["user_requirement"],
        "project_type":        metadata["project_type"],
        "language":            metadata["language"],
        "needs_server":        metadata["needs_server"],
        "needs_compilation":   metadata["needs_compilation"],
        "needs_dependencies":  metadata["needs_dependencies"],
    })

    if not isinstance(decision, OrchestratorDecision):
        # Some providers return a dict instead of a model instance
        decision = OrchestratorDecision.model_validate(decision)

    return decision


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(state: Any) -> Dict[str, Any]:
    """Pull routing-relevant fields with type-safe defaults for the prompt."""
    if not isinstance(state, dict):
        raise TypeError(
            f"Orchestrator expected dict state, got {type(state).__name__}"
        )

    return {
        "user_requirement":   _to_str(state.get("user_requirement"), default=""),
        "needs_server":       _to_bool(state.get("needs_server"), default=False),
        "needs_compilation":  _to_bool(state.get("needs_compilation"), default=False),
        "needs_dependencies": _to_bool(state.get("needs_dependencies"), default=False),
        "project_type":       _to_str(state.get("project_type"), default="unknown"),
        "language":           _to_str(state.get("language"), default="unknown"),
    }


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):  return True
        if v in ("false", "no", "0", ""): return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_str(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned if cleaned else default