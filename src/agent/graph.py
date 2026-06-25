"""
Agent Graph - LangGraph Workflow Orchestration

Defines the complete AI Dev Agent workflow:
1. Planner   → generates plan steps + classification metadata
2. Router    → reads classification, picks the right executor branch
3. Executors → THREE specialized executors handle different project types:
                - SimpleExecutor    (Python scripts, basic CLI tools)
                - CompiledExecutor  (C, C++, Rust, Go, Java)
                - WebExecutor       (FastAPI, Flask, Streamlit)
4. Conditional routing → continue until complete or replanning needed
5. Error recovery → with integrated Error Analyzer and Fix Generator

Day 2 changes from Day 1:
- Three executor nodes instead of one monolithic node
- Router's conditional edges now point to three distinct executors
- Each executor independently routes back via should_continue
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
from src.agent.nodes.simple_executor import simple_executor_node
from src.agent.nodes.compiled_executor import compiled_executor_node
from src.agent.nodes.web_executor import web_executor_node


# ---------------------------------------------------------------------------
# Node name constants — keep in sync with workflow.add_node calls
# ---------------------------------------------------------------------------

NODE_PLANNER = "planner"
NODE_ROUTER = "router"
NODE_SIMPLE_EXEC = "simple_executor"
NODE_COMPILED_EXEC = "compiled_executor"
NODE_WEB_EXEC = "web_executor"

# Maps Router output strings to actual node names in the graph.
# This decoupling means renaming a route doesn't require touching every edge.
ROUTE_TO_NODE = {
    ROUTE_SIMPLE:   NODE_SIMPLE_EXEC,
    ROUTE_COMPILED: NODE_COMPILED_EXEC,
    ROUTE_WEB:      NODE_WEB_EXEC,
}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_to_executor(state: AgentState) -> str:
    """
    Conditional edge from Router → which executor handles this task.

    Router has already populated state["route"]. We translate that into
    a node name. If somehow the route is missing or unknown, fall back to
    the simple executor (safest default — it handles the broadest range).
    """
    route = state.get("route", ROUTE_SIMPLE)
    return ROUTE_TO_NODE.get(route, NODE_SIMPLE_EXEC)


def should_continue(state: AgentState) -> str:
    """
    Conditional edge from any Executor → where to go next.

    Routing logic:
      - is_complete=True       → END (terminal)
      - plan_feedback set      → planner (replan after max retries)
      - otherwise              → re-enter the SAME executor for next step

    Why "same executor": once the Router has classified a project, all
    subsequent steps stay in the same specialized executor. We don't
    re-route mid-execution. This is determined by reading the route from
    state and returning the corresponding executor node name.
    """
    # Terminal: all done
    if state.get("is_complete", False):
        return END

    # Escalation: planner needs to replan after executor exhausted retries
    if state.get("plan_feedback"):
        return NODE_PLANNER

    # Continue: same executor handles the next step
    route = state.get("route", ROUTE_SIMPLE)
    return ROUTE_TO_NODE.get(route, NODE_SIMPLE_EXEC)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_workflow() -> "StateGraph":
    """
    Build, configure, and compile the LangGraph workflow.

    Final architecture (Day 2 complete):

        Planner → Router ─┬─→ SimpleExecutor   ─┐
                          ├─→ CompiledExecutor ─┼─→ (retry / replan / END)
                          └─→ WebExecutor      ─┘

    Each executor loops back to itself for the next step until done.
    On replan, control returns to the Planner for a fresh plan.

    Returns:
        CompiledGraph: Ready-to-use workflow graph
    """
    print("--- Building AI Dev Agent Workflow Graph ---")

    workflow = StateGraph(AgentState)

    # ========== ADD NODES ==========
    print("Adding nodes to graph...")
    workflow.add_node(NODE_PLANNER, planner_node)
    workflow.add_node(NODE_ROUTER, router_node)
    workflow.add_node(NODE_SIMPLE_EXEC, simple_executor_node)
    workflow.add_node(NODE_COMPILED_EXEC, compiled_executor_node)
    workflow.add_node(NODE_WEB_EXEC, web_executor_node)
    print("[+] Added nodes: planner, router, simple_executor, "
          "compiled_executor, web_executor")

    # ========== ADD EDGES ==========
    print("Configuring edges and routing...")

    # Entry point: always start with planner
    workflow.set_entry_point(NODE_PLANNER)

    # Planner → Router (Router reads classification metadata from state)
    workflow.add_edge(NODE_PLANNER, NODE_ROUTER)

    # Router → one of three executors (conditional based on route value)
    workflow.add_conditional_edges(
        NODE_ROUTER,
        route_to_executor,
        {
            NODE_SIMPLE_EXEC:   NODE_SIMPLE_EXEC,
            NODE_COMPILED_EXEC: NODE_COMPILED_EXEC,
            NODE_WEB_EXEC:      NODE_WEB_EXEC,
        },
    )

    # Each executor has its own conditional routing.
    # All three share the same `should_continue` function — it reads
    # state["route"] to figure out which executor to loop back to.
    for executor_node_name in (NODE_SIMPLE_EXEC, NODE_COMPILED_EXEC, NODE_WEB_EXEC):
        workflow.add_conditional_edges(
            executor_node_name,
            should_continue,
            {
                NODE_SIMPLE_EXEC:   NODE_SIMPLE_EXEC,
                NODE_COMPILED_EXEC: NODE_COMPILED_EXEC,
                NODE_WEB_EXEC:      NODE_WEB_EXEC,
                NODE_PLANNER:       NODE_PLANNER,
                END:                END,
            },
        )

    print("[+] Routing configured:")
    print("  - Planner -> Router")
    print("  - Router -> SimpleExecutor | CompiledExecutor | WebExecutor")
    print("  - Each Executor -> Self (next step) | Planner (replan) | END (done)")

    # ========== COMPILE GRAPH ==========
    print("Compiling workflow graph...")
    graph = workflow.compile()
    print("[+] Workflow graph compiled successfully!")

    print("\nWorkflow Summary:")
    print("  Start:    Planner Node")
    print("  Planner:  Generates plan + classification metadata")
    print("            (project_type, language, needs_compilation, needs_server)")
    print("  Router:   Reads metadata, picks the right executor branch")
    print("            Routes: simple | compiled | web")
    print()
    print("  SimpleExecutor:   Python scripts, basic CLI tools")
    print("                    Standard 30s timeout, single-command verification")
    print()
    print("  CompiledExecutor: C, C++, Rust, Go, Java")
    print("                    60s compile timeout, missing-compiler detection")
    print("                    Distinct [compile] vs [run] log phases")
    print()
    print("  WebExecutor:      FastAPI, Flask, Streamlit")
    print("                    180s install timeout, background server launch")
    print("                    HTTP endpoint probing for real verification")
    print()
    print("  All executors:    On failure -> Error Analyzer -> Fix Generator -> Retry")
    print("                    Max retries exceeded -> Feedback to Planner (replan)")
    print("  End:              When all steps complete or unrecoverable failure")

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

    # Test 1: Print graph structure
    print("\nGraph structure:")
    try:
        print(graph)
    except Exception as e:
        print(f"(Graph structure visualization not available: {e})")

    # Test 2: Verify ROUTE_TO_NODE map is complete
    print("\nRoute mapping check:")
    for route, node in ROUTE_TO_NODE.items():
        print(f"  {route!r:12} → {node}")
    print(f"\n  Total routes: {len(ROUTE_TO_NODE)}")

    # Test 3: Confirm route_to_executor handles edge cases
    print("\nrouter_to_executor edge cases:")
    assert route_to_executor({"route": ROUTE_SIMPLE}) == NODE_SIMPLE_EXEC
    assert route_to_executor({"route": ROUTE_COMPILED}) == NODE_COMPILED_EXEC
    assert route_to_executor({"route": ROUTE_WEB}) == NODE_WEB_EXEC
    assert route_to_executor({"route": "unknown_route"}) == NODE_SIMPLE_EXEC
    assert route_to_executor({}) == NODE_SIMPLE_EXEC   # no route → default
    print("  ✓ All known routes map correctly")
    print("  ✓ Unknown route falls back to simple_executor")
    print("  ✓ Missing route falls back to simple_executor")

    # Test 4: Confirm should_continue handles all states
    print("\nshould_continue edge cases:")
    assert should_continue({"is_complete": True}) == END
    assert should_continue({"plan_feedback": "needs work"}) == NODE_PLANNER
    assert should_continue({"route": ROUTE_WEB}) == NODE_WEB_EXEC
    assert should_continue({"route": ROUTE_COMPILED}) == NODE_COMPILED_EXEC
    assert should_continue({}) == NODE_SIMPLE_EXEC  # default loop
    print("  ✓ is_complete=True → END")
    print("  ✓ plan_feedback set → planner")
    print("  ✓ Mid-execution → loops back to same executor by route")

    # Test 5: Mock state for graph readiness check
    print("\n" + "-" * 60)
    print("Mock state structure (ready for graph.invoke())")
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
        # Planner will populate these:
        project_type=None,
        language=None,
        needs_compilation=None,
        needs_server=None,
        needs_dependencies=None,
        # Router will populate this:
        route=None,
    )

    print("\nInitial State:")
    print(f"  Requirement:        {mock_state['requirement']}")
    print(f"  Plan:               {mock_state['plan']}")
    print(f"  Current Step:       {mock_state['current_step']}")
    print(f"  Is Complete:        {mock_state['is_complete']}")
    print(f"  Project Type:       {mock_state.get('project_type')} (set by Planner)")
    print(f"  Route:              {mock_state.get('route')} (set by Router)")

    print("\nExpected flow for this requirement:")
    print("  1. Planner classifies as 'script', sets project_type='script'")
    print("  2. Router reads project_type, sets route='simple'")
    print("  3. simple_executor runs each step")
    print("  4. END when complete")

    print("\n✓ Graph ready for invocation via main.py")