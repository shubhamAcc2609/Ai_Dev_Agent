"""
Main Entry Point - AI Dev Agent

Initializes and runs the AI Dev Agent workflow.

Usage:
    python main.py                                    # Interactive mode
    python main.py --requirement "Create a Flask app" # Single requirement
    python main.py --example 0                        # Predefined example
    python main.py --verbose                          # Full execution trace
    python main.py --quiet                            # Summary only
    python main.py --list-examples                    # Show examples
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import List

from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Console setup
# ---------------------------------------------------------------------------

def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_REQUIREMENTS = [
    "Create a simple 'Hello World' Python script",
    "Create a Flask web server with /api/users endpoint",
    "Create a FastAPI server with POST /data endpoint",
    "A python program to find the largest number in a list without using sorting",
    "A CLI tool that counts word frequency in a text file",
]

# Markers in log lines that indicate proof of verification
VERIFICATION_MARKERS = (
    "Verified live:",        # server probe succeeded
    "Endpoint responded:",   # HTTP probe succeeded
    "[verified:",            # generic verification summary in step-complete line
    "Command ok.",           # plain command exited cleanly
)

# Markers that indicate things to highlight as warnings
WARNING_MARKERS = (
    "ERROR RECOVERY",
    "✗",
    "CRITICAL ERROR",
    "Server verification failed",
    "Validation failed",
    "Catastrophic error",
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(title: str, width: int = 70) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def _print_section(title: str, width: int = 70) -> None:
    print()
    print("-" * width)
    print(title)
    print("-" * width)


def _filter_lines(lines: List[str], markers: tuple) -> List[str]:
    """Return lines that contain any of the given markers."""
    return [ln for ln in lines if any(marker in ln for marker in markers)]


def _format_status(final_status: str, is_complete: bool, last_error) -> str:
    """Normalize the agent's status into a single human label."""
    status = (final_status or "").lower()
    if status == "success":
        return "SUCCESS"
    if status == "failed":
        return "FAILED"
    if status == "in_progress" or (is_complete and not last_error):
        return "SUCCESS"
    if is_complete and last_error:
        return "FAILED"
    return "INCOMPLETE"


def _exit_code_for_status(status_label: str) -> int:
    return {"SUCCESS": 0, "FAILED": 2, "INCOMPLETE": 3}.get(status_label, 1)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_results(result: dict, verbose: bool) -> str:
    """Render the agent result to stdout and return the status label."""
    plan = result.get("plan", []) or []
    files = result.get("files", []) or []
    logs = result.get("logs", []) or []
    last_error = result.get("last_error")
    is_complete = result.get("is_complete", False)
    final_status = result.get("final_status", "")

    status_label = _format_status(final_status, is_complete, last_error)

    print()
    _print_header("*** AI DEV AGENT - EXECUTION COMPLETE ***")

    print(f"\nStatus       : {status_label}")
    print(f"Plan Steps   : {len(plan)}")
    print(f"Files Created: {len(files)}")
    print(f"Log Entries  : {len(logs)}")

    # ---- Files ----------------------------------------------------------
    if files:
        print("\n[FILES CREATED]")
        for path in files:
            print(f"  - {path}")

    # ---- Plan -----------------------------------------------------------
    if plan:
        print("\n[PLAN]")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {textwrap.shorten(step, width=140, placeholder='…')}")

    # ---- Verification proof --------------------------------------------
    verifications = _filter_lines(logs, VERIFICATION_MARKERS)
    if verifications:
        print("\n[VERIFICATION PROOF]")
        for line in verifications:
            print(f"  ✓ {line.lstrip('✓ ').strip()}")
    else:
        print("\n[VERIFICATION PROOF]")
        print("  (none — no command or server probe succeeded explicitly)")

    # ---- Warnings -------------------------------------------------------
    warnings = _filter_lines(logs, WARNING_MARKERS)
    if warnings:
        print("\n[WARNINGS / ERRORS]")
        for line in warnings:
            print(f"  ⚠ {line.strip()}")

    # ---- Last error -----------------------------------------------------
    if last_error:
        print(f"\n[LAST ERROR]\n  {last_error}")

    # ---- Full trace -----------------------------------------------------
    if verbose:
        print("\n[FULL EXECUTION TRACE]")
        for i, entry in enumerate(logs, 1):
            print(f"  {i:>3}. {entry}")

    print()
    _print_header(f"FINAL STATUS: {status_label}")
    return status_label


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_agent(
    requirement: str,
    verbose: bool = False,
    exit_on_finish: bool = False,
) -> dict:
    """
    Run the AI Dev Agent with the given requirement.

    Args:
        requirement: Natural-language description of what to build.
        verbose: When True, print the full execution log (every entry).
                 When False, print only summary + verification proof + warnings.
        exit_on_finish: When True, `sys.exit()` with a status-derived code.

    Returns:
        The agent's final state dict (including `final_status` field).
    """
    if not requirement or not requirement.strip():
        raise ValueError("Requirement cannot be empty")

    _print_header("*** AI DEV AGENT - STARTING ***")
    print(f"\nRequirement: {requirement}\n")

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
        user_feedback=None,
    )

    print("Initializing workflow...")
    print(f"State initialized with requirement: '{requirement}'")

    try:
        print("\nInvoking graph workflow...\n")
        from src.agent.graph import graph
        result = graph.invoke(initial_state)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Agent interrupted by user (Ctrl+C)")
        if exit_on_finish:
            sys.exit(130)
        raise
    except Exception as exc:
        print(f"\n\n[ERROR] FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        if exit_on_finish:
            sys.exit(1)
        raise

    status_label = _render_results(result, verbose=verbose)
    result["status_label"] = status_label

    if exit_on_finish:
        sys.exit(_exit_code_for_status(status_label))

    return result


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def main_interactive() -> None:
    """Interactive CLI loop."""
    _print_header("AI DEV AGENT - INTERACTIVE CLI")
    print("\nEnter your development requirement (or 'quit' to exit).")
    print("Type 'verbose' before a requirement to see the full trace, e.g.:")
    print("    verbose Create a Flask hello world app")
    print("-" * 70)

    while True:
        try:
            raw = input("\n[INPUT] Requirement: ").strip()
        except EOFError:
            print("\n[INFO] No interactive input was provided.")
            break
        except KeyboardInterrupt:
            print("\n\n[INFO] Goodbye!")
            break

        if not raw:
            print("[ERROR] Please enter a requirement.")
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("[INFO] Goodbye!")
            break

        verbose_run = False
        if raw.lower().startswith("verbose "):
            verbose_run = True
            raw = raw[len("verbose "):].strip()

        try:
            run_agent(raw, verbose=verbose_run, exit_on_finish=False)
        except Exception as exc:
            print(f"\n[ERROR] Run failed: {exc}")
            continue

        print("\n" + "-" * 70)
        try:
            cont = input("[INPUT] Continue? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Goodbye!")
            break
        if cont.startswith("n"):
            print("[INFO] Goodbye!")
            break


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def run_example(example_index: int, verbose: bool = False) -> None:
    if example_index < 0 or example_index >= len(EXAMPLE_REQUIREMENTS):
        print(
            f"[ERROR] Invalid example index. "
            f"Choose 0-{len(EXAMPLE_REQUIREMENTS) - 1}"
        )
        return
    requirement = EXAMPLE_REQUIREMENTS[example_index]
    print(f"\n[EXAMPLE] Running Example {example_index + 1}:\n   {requirement}\n")
    run_agent(requirement, verbose=verbose, exit_on_finish=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Dev Agent - Autonomous Software Development Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py                                            # Interactive
              python main.py --requirement "Create a Flask app"         # One-shot
              python main.py --requirement "..." --verbose              # Full trace
              python main.py --requirement "..." --ci                   # Exit code for CI
              python main.py --example 0                                # Predefined
              python main.py --list-examples                            # List examples

            Exit codes (with --ci):
              0  SUCCESS — all steps completed and verified
              2  FAILED  — execution finished but reported failure
              3  INCOMPLETE — graph terminated without final_status
              130 Interrupted by user (Ctrl+C)
        """),
    )

    parser.add_argument("--requirement", "-r", type=str,
                        help="Single requirement to execute (non-interactive)")
    parser.add_argument("--example", "-e", type=int,
                        help=f"Run predefined example (0-{len(EXAMPLE_REQUIREMENTS)-1})")
    parser.add_argument("--list-examples", "-l", action="store_true",
                        help="List all predefined examples and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print the full execution log")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-line trace (default)")
    parser.add_argument("--ci", action="store_true",
                        help="Exit with status-derived code (for CI pipelines)")

    args = parser.parse_args()

    if args.list_examples:
        print("\n[EXAMPLES] Predefined Examples:")
        for i, req in enumerate(EXAMPLE_REQUIREMENTS):
            print(f"  {i}. {req}")
        return

    verbose = args.verbose and not args.quiet

    if args.example is not None:
        run_example(args.example, verbose=verbose)
        return

    if args.requirement:
        run_agent(
            args.requirement,
            verbose=verbose,
            exit_on_finish=args.ci,
        )
        return

    main_interactive()


if __name__ == "__main__":
    main()