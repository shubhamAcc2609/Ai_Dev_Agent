"""
Main Entry Point - AI Dev Agent

Initializes and runs the AI Dev Agent workflow.

Usage:
    python main.py
    
    Or import and use:
    from main import run_agent
    result = run_agent("Create a FastAPI server")
"""

import sys
from src.agent.state import AgentState


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()


def run_agent(requirement: str, verbose: bool = True) -> dict:
    """
    Runs the AI Dev Agent with the given requirement.
    
    Workflow:
    1. Initialize state with requirement
    2. Invoke graph (Planner → Executor loop)
    3. Return final state with results, logs, and files
    
    Args:
        requirement: Natural language description of what to build/do
        verbose: Whether to print detailed execution logs
    
    Returns:
        dict: Final agent state with:
            - plan: Generated steps
            - files: Created files
            - logs: Execution logs
            - is_complete: Whether task completed
            - last_error: Any final error (if failed)
    
    Example:
        result = run_agent("Create a Python Flask app with /hello endpoint")
        print(f"Files created: {result['files']}")
        print(f"Success: {result['is_complete']}")
    """
    
    if not requirement or not requirement.strip():
        raise ValueError("Requirement cannot be empty")
    
    print("="*70)
    print("*** AI DEV AGENT - STARTING ***")
    print("="*70)
    print(f"\nRequirement: {requirement}\n")
    
    # ========== INITIALIZE STATE ==========
    initial_state = AgentState(
        requirement=requirement,
        plan=[],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False,
        last_error=None,
        retry_count=0,
        plan_feedback=None,
        user_feedback=None
    )
    
    print("Initializing workflow...")
    print(f"State initialized with requirement: '{requirement}'")
    
    # ========== INVOKE GRAPH ==========
    try:
        print("\nInvoking graph workflow...\n")
        from src.agent.graph import graph
        
        result = graph.invoke(initial_state)
        
        # ========== PRINT RESULTS ==========
        print("\n" + "="*70)
        print("*** AI DEV AGENT - EXECUTION COMPLETE ***")
        print("="*70)
        
        is_complete = result.get("is_complete", False)
        plan = result.get("plan", [])
        files = result.get("files", [])
        #logs = result.get("logs", [])
        last_error = result.get("last_error")
        
        print(f"\nStatus: {'SUCCESS' if is_complete else 'INCOMPLETE'}")
        print(f"Plan Steps: {len(plan)}")
        print(f"Files Created: {len(files)}")
       #print(f"Logs: {len(logs)} entries")
        
        if files:
            print("\n[FILES CREATED]:")
            for file_path in files:
                print(f"  - {file_path}")
        
        if last_error:
            print(f"\n[WARNING] Last Error: {last_error}")
        
        #if verbose and logs:
           # print("\n[EXECUTION LOGS]:")
           # for i, log in enumerate(logs, 1):
           #     print(f"  {i}. {log}")
        
        print("\n" + "="*70)
        
        return result
    
    except KeyboardInterrupt:
        print("\n\n[WARNING] Agent interrupted by user (Ctrl+C)")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n[ERROR] FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Interactive CLI for the AI Dev Agent.
    """
    print("\n" + "="*70)
    print("AI DEV AGENT - INTERACTIVE CLI")
    print("="*70)
    print("\nEnter your development requirement (or 'quit' to exit):")
    print("-"*70)
    
    while True:
        try:
            requirement = input("\n[INPUT] Requirement: ").strip()
        except EOFError:
            print("\n[INFO] No interactive input was provided.")
            print('[INFO] Run with --requirement "Create a Hello World Python script" for non-interactive mode.')
            break
        except KeyboardInterrupt:
            print("\n\n[INFO] Goodbye!")
            break

        if requirement.lower() in ("quit", "exit", "q"):
            print("[INFO] Goodbye!")
            break

        if not requirement:
            print("[ERROR] Please enter a requirement")
            continue

        run_agent(requirement, verbose=True)

        print("\n" + "-"*70)
        try:
            user_input = input("[INPUT] Continue? (yes/no/modify): ").strip().lower()
        except EOFError:
            print("\n[INFO] No follow-up input was provided. Exiting.")
            break
        except KeyboardInterrupt:
            print("\n\n[INFO] Goodbye!")
            break

        if user_input.startswith("no"):
            break
        elif user_input.startswith("mod"):
            print("\n[INFO] Modifying state for next iteration...")
            # Could add feedback logic here


# ========== EXAMPLE REQUIREMENTS ==========
EXAMPLE_REQUIREMENTS = [
    "Create a simple 'Hello World' Python script",
    "Create a Flask web server with /api/users endpoint",
    "Create a FastAPI server with POST /data endpoint",
    "Create a requirements.txt with common Python packages",
    "Create a README.md with project documentation",
]


def run_example(example_index: int = 0):
    """
    Runs a predefined example.
    
    Args:
        example_index: Index of example (0-4)
    """
    if example_index < 0 or example_index >= len(EXAMPLE_REQUIREMENTS):
        print(f"[ERROR] Invalid example index. Choose 0-{len(EXAMPLE_REQUIREMENTS)-1}")
        return
    
    requirement = EXAMPLE_REQUIREMENTS[example_index]
    print(f"\n[EXAMPLE] Running Example {example_index + 1}:")
    print(f"   {requirement}\n")
    
    run_agent(requirement, verbose=True)


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Dev Agent - Autonomous Software Development Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --requirement "Create a Flask app" # Single requirement
  python main.py --example 0                        # Run predefined example
  python main.py --list-examples                    # Show all examples
        """
    )
    
    parser.add_argument(
        "--requirement", "-r",
        type=str,
        help="Single requirement to execute (non-interactive)"
    )
    
    parser.add_argument(
        "--example", "-e",
        type=int,
        help="Run predefined example by index (0-4)"
    )
    
    parser.add_argument(
        "--list-examples", "-l",
        action="store_true",
        help="List all predefined examples"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print detailed execution logs"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list_examples:
        print("\n[EXAMPLES] Predefined Examples:")
        for i, req in enumerate(EXAMPLE_REQUIREMENTS):
            print(f"  {i}. {req}")
    
    elif args.example is not None:
        run_example(args.example)
    
    elif args.requirement:
        run_agent(args.requirement, verbose=args.verbose)
    
    else:
        # Interactive mode
        main()
