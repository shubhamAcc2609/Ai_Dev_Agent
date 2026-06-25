"""
Agent Graph - LangGraph Workflow Orchestration

Defines the complete AI Dev Agent workflow:
1. Planner   → generates plan steps + classification metadata
2. Router    → reads classification, picks the right executor branch
3. Executor  → orchestrates Code Generator, File Manager, Execution Manager
4. Conditional routing → continue until complete or replanning needed
5. Error recovery → with integrated Error Analyzer and Fix Generator
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.planner import planner_node
from src.agent.nodes.router import (
    router_node,
    ROUTE_SIMPLE,
    ROUTE_COMPILED,
    ROUTE_WEB,
)
from src.agent.nodes.executor import executor_node


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_to_executor(state: AgentState) -> str:
    """
    Conditional edge from Router → which executor handles this task.

    The Router has already populated `state["route"]` based on the Planner's
    classification metadata. This function just translates the route string
    into the next graph node name.

    DAY 1 NOTE: All three routes currently map to the same "executor" node
    because the executor split hasn't happened yet. On Day 2 this map will
    point to "simple_executor" / "compiled_executor" / "web_executor".

    Returns:
        str: Name of the next graph node to invoke.
    """
    route = state.get("route", ROUTE_SIMPLE)

    # Day 1: all three routes go to the existing executor.
    # The architecture is visible (3 routes!) but behavior is unchanged.
    # On Day 2, change this map to the three specialized executors.
    return {
        ROUTE_SIMPLE:   "executor",
        ROUTE_COMPILED: "executor",
        ROUTE_WEB:      "executor",
    }.get(route, "executor")  # Safe default — always returns a valid node


def should_continue(state: AgentState) -> str:
    """
    Conditional edge from Executor → where to go next.

    Routing logic:
      - is_complete=True → END
      - plan_feedback set → back to Planner for replanning
      - otherwise → re-enter executor for the next step

    Args:
        state: Current agent state

    Returns:
        str: Next node name ("executor", "planner", or END)
    """
    # If agent marked complete, terminate
    if state.get("is_complete", False):
        return END

    # If planner feedback is set, replan (executor exhausted retries)
    if state.get("plan_feedback"):
        return "planner"

    # Otherwise, keep executing — executor manages step progression internally
    return "executor"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_workflow() -> "StateGraph":
    """
    Build, configure, and compile the LangGraph workflow.

    Architecture (Day 1):
        Planner → Router → Executor ⟲ (with replan loop back to Planner)

    Architecture (Day 2 target):
        Planner → Router → ┬─→ SimpleExecutor   ─┐
                            ├─→ CompiledExecutor ─┼─→ END (or replan)
                            └─→ WebExecutor      ─┘

    Returns:
        CompiledGraph: Ready-to-use workflow graph
    """
    print("--- Building AI Dev Agent Workflow Graph ---")

    workflow = StateGraph(AgentState)

    # ========== ADD NODES ==========
    print("Adding nodes to graph...")
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("executor", executor_node)
    print("[+] Added nodes: planner, router, executor")

    # ========== ADD EDGES ==========
    print("Configuring edges and routing...")

    # Entry point: always start with planner
    workflow.set_entry_point("planner")

    # Planner → Router (Router reads classification metadata from state)
    workflow.add_edge("planner", "router")

    # Router → Executor (conditional based on route value)
    workflow.add_conditional_edges(
        "router",
        route_to_executor,
        {
            "executor": "executor",
            # Day 2 target (uncomment when executors are split):
            # "simple_executor":   "simple_executor",
            # "compiled_executor": "compiled_executor",
            # "web_executor":      "web_executor",
        },
    )

    # Executor → conditional routing (retry, replan, or end)
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",  # Retry / next step
            "planner":  "planner",   # Replan (after max retries)
            END:        END,         # All done
        },
    )

    print("[+] Routing configured:")
    print("  - Planner -> Router")
    print("  - Router -> Executor (by route: simple | compiled | web)")
    print("  - Executor -> Executor (retry) | Planner (replan) | END (complete)")

    # ========== COMPILE GRAPH ==========
    print("Compiling workflow graph...")
    graph = workflow.compile()
    print("[+] Workflow graph compiled successfully!")

    print("\nWorkflow Summary:")
    print("  Start: Planner Node")
    print("  Planner: Generates plan + classification (project_type, language, etc.)")
    print("  Router:  Reads classification, picks the right executor branch")
    print("           Routes: simple | compiled | web")
    print("  Executor: Runs each step (Code Generator -> File Manager -> Execution Manager)")
    print("            On failure -> Error Analyzer -> Fix Generator -> Retry")
    print("            Max retries exceeded -> Feedback to Planner (replan)")
    print("  End: When all steps complete or user intervention needed")

    return graph


# ---------------------------------------------------------------------------
# Global workflow instance
# ---------------------------------------------------------------------------

graph = create_workflow()


# ---------------------------------------------------------------------------
# Local testing block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING GRAPH STRUCTURE")
    print("=" * 60)

    # Test: Print graph structure
    print("\nGraph structure:")
    try:
        print(graph)
    except Exception as e:
        print(f"(Graph structure visualization not available: {e})")

    # Test: Create mock state and confirm graph is invokable
    print("\n" + "-" * 60)
    print("Testing graph execution with mock requirement")
    print("-" * 60)

    mock_state = AgentState(
        requirement="Create a simple 'Hello World' Python script",
        plan=[],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False,
        last_error=None,
        retry_count=0,
        plan_feedback=None,
        user_feedback=None,
        # New fields populated by Planner/Router (defaults shown):
        project_type=None,
        language=None,
        needs_compilation=None,
        needs_server=None,
        needs_dependencies=None,
        route=None,
    )

    print("\nInitial State:")
    print(f"  Requirement:       {mock_state['requirement']}")
    print(f"  Plan:              {mock_state['plan']}")
    print(f"  Current Step:      {mock_state['current_step']}")
    print(f"  Is Complete:       {mock_state['is_complete']}")
    print(f"  Route (pre-router): {mock_state.get('route')}")

    print("\n(Graph ready for invocation in main.py)")
    print("  Expected flow on invoke:")
    print("    1. Planner generates plan + sets project_type='script'")
    print("    2. Router reads project_type, sets route='simple'")
    print("    3. Executor runs each step")
    print("    4. END when complete\n")