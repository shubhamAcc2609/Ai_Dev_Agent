"""
Agent State

Shared state object passed between all nodes in the LangGraph workflow.

Each node reads what it needs and returns a delta (partial update).
LangGraph merges the delta into the canonical state.

Fields fall into 3 categories:
- Inputs:     requirement, user_feedback
- Planner:    plan, project_type, language, needs_compilation, needs_server
- Execution:  current_step, files, logs, retry_count, last_error
- Control:    is_complete, plan_feedback, route, final_status
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Global state shared across all LangGraph nodes.

    Using total=False means every field is optional at the schema level —
    individual nodes can return partial dicts without TypedDict complaining.
    """

    # ─── Inputs ──────────────────────────────────────────────────────────
    requirement: str                # The user's natural-language requirement
    user_feedback: Optional[str]    # Manual feedback from human-in-the-loop

    # ─── Planner outputs ─────────────────────────────────────────────────
    plan: List[str]                          # Ordered list of step descriptions
    project_type: Optional[str]              # "script" | "utility" | "application" | "compiled"
    language: Optional[str]                  # "python" | "cpp" | "rust" | "javascript" | ...
    needs_compilation: Optional[bool]        # True if step needs gcc/rustc/etc.
    needs_server: Optional[bool]             # True if step launches a long-running server
    needs_dependencies: Optional[bool]       # True if pip install / npm install is required
    plan_feedback: Optional[str]             # Feedback when a step asks for replanning

    # ─── Router output ───────────────────────────────────────────────────
    route: Optional[str]            # "simple" | "compiled" | "web"

    # ─── Execution state ─────────────────────────────────────────────────
    current_step: int               # Index of next step to execute
    files: List[str]                # Files created/modified by executor
    logs: List[str]                 # Human-readable execution trace
    retry_count: int                # How many times current step has been retried
    last_error: Optional[str]       # Last error message (cleared on success)

    # ─── Control flags ───────────────────────────────────────────────────
    is_complete: bool               # True when graph should terminate
    final_status: Optional[str]     # "success" | "failed" | "in_progress"