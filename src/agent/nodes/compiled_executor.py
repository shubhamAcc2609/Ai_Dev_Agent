"""
Compiled Executor

Handles compiled languages: C, C++, Rust, Go, Java, C#.

Characteristics:
- Two-stage execution: compile, then run
- Longer timeout for compile commands (compilers can be slow)
- Missing-compiler detection with actionable error messages
- Verification = compiled binary runs and exits with code 0

Bulletproof guarantees:
- Never raises — all exceptions caught and translated to failure deltas
- Defensive against malformed state (handled by shared helpers)
- Defensive against malformed code-generator output (validated locally)
- Defensive against execute_command crashes (caught and wrapped)
- Defensive against file_manager crashes (caught and wrapped)
- Returns valid state delta in every code path

Specialization from simple_executor:
- Detects compile commands via _looks_like_compile()
- Splits timeout: compile gets 60s, run gets 30s
- Recognizes missing-compiler errors via _is_missing_compiler()
- Surfaces compile-vs-run in the verification_summary
"""

from __future__ import annotations

import logging
from typing import List

from src.agent.nodes._executor_shared import (
    STATUS_FAILED,
    StepResult,
    build_success_delta,
    check_all_steps_done,
    check_empty_plan,
    extract_state_basics,
    generate_plan_for_step,
    handle_step_failure,
    make_delta,
    make_logger,
    write_file_if_needed,
)
from src.agent.state import AgentState
from src.tools.execution_manager import execute_command

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# This executor's specific tuning knobs
# ---------------------------------------------------------------------------

COMPILE_TIMEOUT = 60         # Compilers can take a while on bigger programs
RUN_TIMEOUT = 30             # Compiled binaries should be fast to run
MAX_OUTPUT_PREVIEW = 200     # Chars of stdout/stderr to log per command

# Operations recognized in code-generator output
OP_CREATE_FILE = "create_file"
OP_UPDATE_FILE = "update_file"
OP_RUN_COMMAND = "execute_command"
OP_VERIFY = "verify"
VALID_OPERATIONS = {OP_CREATE_FILE, OP_UPDATE_FILE, OP_RUN_COMMAND, OP_VERIFY}

# Substrings (case-insensitive) that signal a compile command.
# Order matters only for readability — we check membership, not priority.
COMPILE_HINTS = (
    "gcc", "g++", "clang", "clang++", "cc ", "c++",
    "rustc", "cargo build",
    "go build",
    "javac",
    "csc ", "dotnet build",
    "make ", "cmake ", "ninja",
)

# Substrings that signal "compiler not installed" in stderr.
# We surface a helpful hint instead of treating these as opaque failures.
MISSING_COMPILER_SIGNALS = (
    "is not recognized as an internal or external command",   # Windows cmd
    "is not recognized as the name of a cmdlet",              # PowerShell
    "command not found",                                       # POSIX shells
    "no such file or directory: 'gcc'",                        # macOS clang missing
    "no such file or directory: 'g++'",
    "no such file or directory: 'clang'",
    "no such file or directory: 'rustc'",
    "'cargo' is not recognized",
    "'go' is not recognized",
    "'javac' is not recognized",
)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def compiled_executor_node(state: AgentState) -> dict:
    """
    Execute a single step for a compiled-language project (C, C++, Rust, Go, Java).

    NEVER RAISES. Every failure mode produces a valid state delta.
    """
    # ─── Set up logging that feeds both stdout and state delta ──────────
    new_logs: List[str] = []
    log = make_logger(new_logs)
    log("--- COMPILED EXECUTOR ---")

    # ─── Extract state defensively ──────────────────────────────────────
    try:
        s = extract_state_basics(state)
    except Exception as exc:
        logger.exception("State extraction failed catastrophically")
        log(f"CRITICAL: state extraction failed: {exc}", level=logging.ERROR)
        return make_delta(
            logs=new_logs,
            is_complete=True,
            final_status=STATUS_FAILED,
            last_error=f"State extraction failed: {exc}",
        )

    # ─── Terminal-condition guards ──────────────────────────────────────
    delta = check_empty_plan(s, new_logs, log)
    if delta is not None:
        return delta

    delta = check_all_steps_done(s, new_logs, log)
    if delta is not None:
        return delta

    # ─── Execute the current step ───────────────────────────────────────
    step_description = s["plan"][s["current_step"]]
    log(f"Step {s['current_step'] + 1}/{len(s['plan'])}: {step_description}")

    try:
        result = _run_compiled_step(
            step_description=step_description,
            requirement=s["requirement"],
            plan=s["plan"],
            files_so_far=s["existing_files"],
            log=log,
        )
    except Exception as exc:
        logger.exception("Compiled executor crashed during step execution")
        log(f"CRITICAL ERROR: {exc}", level=logging.ERROR)
        return handle_step_failure(
            base_logs=s["existing_logs"] + new_logs,
            base_files=s["existing_files"],
            retry_count=s["retry_count"],
            current_step=s["current_step"],
            total_steps=len(s["plan"]),
            step_description=step_description,
            error_msg=str(exc),
            stdout="",
            stderr=str(exc),
            command="<executor crashed>",
            file_path="",
            file_content="",
        )

    # ─── Final defensive guard: result must be a StepResult ─────────────
    if not isinstance(result, StepResult):
        msg = (
            f"_run_compiled_step returned {type(result).__name__} "
            f"instead of StepResult"
        )
        logger.error(msg)
        log(f"CRITICAL: {msg}", level=logging.ERROR)
        return handle_step_failure(
            base_logs=s["existing_logs"] + new_logs,
            base_files=s["existing_files"],
            retry_count=s["retry_count"],
            current_step=s["current_step"],
            total_steps=len(s["plan"]),
            step_description=step_description,
            error_msg=msg,
            stdout="",
            stderr=msg,
            command="<returned wrong type>",
            file_path="",
            file_content="",
        )

    # ─── Branch on success or failure ───────────────────────────────────
    if result.success:
        return build_success_delta(s, result, new_logs, log)

    return handle_step_failure(
        base_logs=s["existing_logs"] + new_logs,
        base_files=s["existing_files"] + result.files_created,
        retry_count=s["retry_count"],
        current_step=s["current_step"],
        total_steps=len(s["plan"]),
        step_description=step_description,
        error_msg=result.error_message,
        stdout=result.stdout,
        stderr=result.stderr,
        command=result.command,
        file_path=result.file_path,
        file_content=result.file_content,
    )


# ---------------------------------------------------------------------------
# Specialized step runner — what makes this executor "compiled"
# ---------------------------------------------------------------------------

def _run_compiled_step(
    *,
    step_description: str,
    requirement: str,
    plan: List[str],
    files_so_far: List[str],
    log,
) -> StepResult:
    """
    Write source, then compile or run with mode-aware timeouts.

    Two key behaviors that differ from simple_executor:
    1. If command looks like a compile invocation, use COMPILE_TIMEOUT
    2. If compile fails with "compiler not found" pattern, surface a
       helpful hint instead of dumping raw stderr

    GUARANTEE: Always returns a StepResult, never raises.
    """
    result = StepResult()

    # ─── Step 1: Generate the execution plan via code-generator ─────────
    log("[plan] generating execution plan...")
    try:
        plan_dict = generate_plan_for_step(
            step_description, requirement, plan, files_so_far,
        )
    except Exception as exc:
        log(f"[plan] ✗ code-generator failed: {exc}", level=logging.ERROR)
        result.error_message = f"Code generator failed: {exc}"
        result.stderr = str(exc)
        return result

    # ─── Step 2: Validate plan structure defensively ────────────────────
    if not isinstance(plan_dict, dict):
        msg = f"Code generator returned {type(plan_dict).__name__}, expected dict"
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    operation = _safe_get_str(plan_dict, "operation")
    file_path = _safe_get_str(plan_dict, "file_path")
    file_content = _safe_get_str(plan_dict, "file_content", default="")
    command = _safe_get_str(plan_dict, "command")

    result.plan = plan_dict
    result.command = command
    result.file_path = file_path
    result.file_content = file_content

    log(
        f"[plan] operation='{operation}', file_path='{file_path or 'none'}', "
        f"has_command={bool(command)}"
    )

    # ─── Step 3: Operation-specific handling ────────────────────────────

    if operation == OP_VERIFY:
        log("[verify] no-op step (logical checkpoint)")
        result.success = True
        result.verification_summary = "verify step (no action required)"
        return result

    if operation and operation not in VALID_OPERATIONS:
        msg = (
            f"Unknown operation '{operation}'; expected one of "
            f"{sorted(VALID_OPERATIONS)}"
        )
        log(f"[plan] ✗ {msg}", level=logging.ERROR)
        result.error_message = msg
        result.stderr = msg
        return result

    # ─── Step 4: Write the file (if requested) ──────────────────────────
    try:
        if not write_file_if_needed(file_path, file_content, result, log):
            return result
    except Exception as exc:
        log(f"[file] ✗ file write crashed: {exc}", level=logging.ERROR)
        result.error_message = f"File write crashed: {exc}"
        result.stderr = str(exc)
        return result

    # ─── Step 5: Execute the command (if requested) ─────────────────────
    if not command:
        log("[run] skipped (no command)")
        result.success = True
        return result

    is_compile = _looks_like_compile(command)
    timeout = COMPILE_TIMEOUT if is_compile else RUN_TIMEOUT
    mode = "compile" if is_compile else "run"
    log(f"[{mode}] {command}  (timeout={timeout}s)")

    try:
        ok, stdout, stderr = execute_command(command, timeout=timeout)
    except Exception as exc:
        log(f"[{mode}] ✗ execute_command crashed: {exc}", level=logging.ERROR)
        result.error_message = f"Command execution crashed: {exc}"
        result.stderr = str(exc)
        return result

    # Always capture output for failure analysis even on success
    result.stdout = stdout or ""
    result.stderr = stderr or ""

    if not ok:
        # Special-case: missing compiler → produce actionable hint
        if _is_missing_compiler(stderr):
            tool = _guess_missing_tool(command, stderr)
            hint = (
                f"Compiler '{tool}' not found on host. "
                f"Install it (e.g. via MSYS2: `pacman -S mingw-w64-ucrt-x86_64-gcc`) "
                f"or have the planner switch to an interpreted language like Python."
            )
            log(f"[{mode}] ✗ Missing compiler: {tool!r}", level=logging.ERROR)
            result.error_message = f"Missing compiler ({tool}): {hint}"
            return result

        # Generic command failure
        preview = (stderr or stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
        log(f"[{mode}] ✗ failed: {preview!r}", level=logging.ERROR)
        result.error_message = f"{mode.title()} failed: {preview}"
        return result

    stdout_preview = (stdout or "").strip()[:MAX_OUTPUT_PREVIEW]
    log(f"[{mode}] ✓ exit=0  stdout[:120]={stdout_preview[:120]!r}")

    result.verification_summary = (
        f"{mode}d OK; stdout[:80]={stdout_preview[:80]!r}"
    )
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Compile / missing-compiler heuristics
# ---------------------------------------------------------------------------

def _looks_like_compile(command: str) -> bool:
    """True if the command appears to invoke a compiler or build tool."""
    if not command:
        return False
    cmd_lower = command.lower()
    return any(hint in cmd_lower for hint in COMPILE_HINTS)


def _is_missing_compiler(stderr: str) -> bool:
    """
    True if stderr matches a known 'compiler not installed' pattern.

    This lets us surface a helpful hint instead of a confusing
    'command failed' message when the user's host is missing g++/rustc/etc.
    """
    if not stderr:
        return False
    err_lower = stderr.lower()
    return any(signal in err_lower for signal in MISSING_COMPILER_SIGNALS)


def _guess_missing_tool(command: str, stderr: str) -> str:
    """
    Best-effort: identify WHICH tool is missing.

    Checks the first token of the command against known compilers.
    Falls back to 'the compiler' if nothing matches.
    """
    if not command:
        return "the compiler"

    first_token = command.strip().split()[0].lower()
    known_tools = {
        "gcc", "g++", "clang", "clang++", "cc",
        "rustc", "cargo",
        "go",
        "javac", "java",
        "csc", "dotnet",
        "make", "cmake", "ninja",
    }
    if first_token in known_tools:
        return first_token

    # Try to extract from stderr (e.g., "'g++' is not recognized")
    for tool in known_tools:
        if f"'{tool}'" in stderr.lower() or f"`{tool}`" in stderr.lower():
            return tool

    return "the compiler"


# ---------------------------------------------------------------------------
# Defensive accessor for code-generator output
# ---------------------------------------------------------------------------

def _safe_get_str(d: dict, key: str, default: str = "") -> str:
    """
    Get a key from a dict, coerce to stripped str, fall back to default.

    Handles None, numbers, bools, nested objects — LLM outputs aren't
    always cleanly typed.
    """
    value = d.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


# ---------------------------------------------------------------------------
# Self-test (structural, no LLM calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Compiled Executor (structural tests, no LLM calls)...\n")

    # ───────────────────────────────────────────────────────────────
    # Test 1: _looks_like_compile detects compile commands
    # ───────────────────────────────────────────────────────────────
    assert _looks_like_compile("gcc main.c -o main")
    assert _looks_like_compile("g++ -std=c++17 main.cpp -o app")
    assert _looks_like_compile("clang main.c -o main")
    assert _looks_like_compile("rustc main.rs")
    assert _looks_like_compile("cargo build --release")
    assert _looks_like_compile("go build -o app main.go")
    assert _looks_like_compile("javac Main.java")
    assert _looks_like_compile("make all")
    assert _looks_like_compile("cmake --build .")

    # Negative cases
    assert not _looks_like_compile("./main")
    assert not _looks_like_compile("python main.py")
    assert not _looks_like_compile("./app arg1 arg2")
    assert not _looks_like_compile("")
    assert not _looks_like_compile("java -jar Main.jar")    # `java` alone isn't compile
    print("✓ _looks_like_compile distinguishes compile vs run commands")

    # ───────────────────────────────────────────────────────────────
    # Test 2: _is_missing_compiler detects common patterns
    # ───────────────────────────────────────────────────────────────
    assert _is_missing_compiler(
        "'g++' is not recognized as an internal or external command,\noperable program or batch file."
    )
    assert _is_missing_compiler("bash: gcc: command not found")
    assert _is_missing_compiler("zsh: no such file or directory: 'clang'")
    assert _is_missing_compiler("'cargo' is not recognized")

    # Negative cases
    assert not _is_missing_compiler("undefined reference to `main`")
    assert not _is_missing_compiler("error: expected ';' before '}'")
    assert not _is_missing_compiler("")
    assert not _is_missing_compiler(None or "")
    print("✓ _is_missing_compiler recognizes missing-compiler patterns")

    # ───────────────────────────────────────────────────────────────
    # Test 3: _guess_missing_tool identifies the right tool
    # ───────────────────────────────────────────────────────────────
    assert _guess_missing_tool("g++ main.cpp", "") == "g++"
    assert _guess_missing_tool("gcc main.c -o main", "") == "gcc"
    assert _guess_missing_tool("cargo build", "") == "cargo"
    assert _guess_missing_tool("javac Main.java", "") == "javac"
    assert _guess_missing_tool("./main", "") == "the compiler"
    assert _guess_missing_tool("", "") == "the compiler"

    # Extract from stderr when command doesn't help
    assert _guess_missing_tool("script.sh", "'g++' is not recognized") == "g++"
    print("✓ _guess_missing_tool identifies tool from command or stderr")

    # ───────────────────────────────────────────────────────────────
    # Test 4: _safe_get_str handles LLM output oddities
    # ───────────────────────────────────────────────────────────────
    assert _safe_get_str({"k": "value"}, "k") == "value"
    assert _safe_get_str({"k": None}, "k") == ""
    assert _safe_get_str({"k": 42}, "k") == "42"
    assert _safe_get_str({}, "missing", default="fallback") == "fallback"
    print("✓ _safe_get_str handles None, int, missing keys")

    # ───────────────────────────────────────────────────────────────
    # Test 5: VALID_OPERATIONS set is correct
    # ───────────────────────────────────────────────────────────────
    assert OP_CREATE_FILE in VALID_OPERATIONS
    assert OP_RUN_COMMAND in VALID_OPERATIONS
    assert OP_VERIFY in VALID_OPERATIONS
    assert "format_drive" not in VALID_OPERATIONS
    print("✓ VALID_OPERATIONS contains expected operations")

    # ───────────────────────────────────────────────────────────────
    # Test 6: compiled_executor_node handles empty plan
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": [],
        "current_step": 0,
        "logs": [],
        "files": [],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = compiled_executor_node(state)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    assert delta.get("final_status") == STATUS_FAILED
    print("✓ empty plan → terminal failure delta")

    # ───────────────────────────────────────────────────────────────
    # Test 7: compiled_executor_node handles all-steps-done
    # ───────────────────────────────────────────────────────────────
    state = {
        "plan": ["step1", "step2"],
        "current_step": 2,
        "logs": [],
        "files": ["main.c", "main"],
        "retry_count": 0,
        "requirement": "test",
    }
    delta = compiled_executor_node(state)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    print("✓ all-steps-done → terminal delta")

    # ───────────────────────────────────────────────────────────────
    # Test 8: compiled_executor_node handles None / non-dict state
    # ───────────────────────────────────────────────────────────────
    delta = compiled_executor_node(None)
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True

    delta = compiled_executor_node("not a state")
    assert isinstance(delta, dict)
    assert delta.get("is_complete") is True
    print("✓ None / non-dict state → graceful failure (no crash)")

    # ───────────────────────────────────────────────────────────────
    # Test 9: timeout selection logic
    # ───────────────────────────────────────────────────────────────
    # We can verify _looks_like_compile drives the timeout choice
    # without actually running commands
    assert COMPILE_TIMEOUT > RUN_TIMEOUT, "Compile timeout should be longer than run"
    assert _looks_like_compile("gcc main.c") is True
    assert _looks_like_compile("./main") is False
    print(f"✓ timeout logic: compile={COMPILE_TIMEOUT}s, run={RUN_TIMEOUT}s")

    # ───────────────────────────────────────────────────────────────
    # Test 10: StepResult contract — _run_compiled_step never raises
    # ───────────────────────────────────────────────────────────────
    new_logs = []
    log = make_logger(new_logs)
    try:
        result = _run_compiled_step(
            step_description="",
            requirement="test",
            plan=[""],
            files_so_far=[],
            log=log,
        )
        assert isinstance(result, StepResult), \
            f"_run_compiled_step returned {type(result).__name__} not StepResult"
        print("✓ _run_compiled_step returns StepResult (even on edge cases)")
    except Exception as exc:
        raise AssertionError(
            f"_run_compiled_step raised instead of returning StepResult: {exc}"
        )

    print("\n✓ All 10 structural self-tests passed.")
    print("\nNote: Integration testing (real compile + run) happens when you")
    print("run `python main.py` with a real C/C++/Rust prompt.")