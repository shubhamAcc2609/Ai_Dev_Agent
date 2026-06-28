"""
AI Dev Agent — Futuristic Streamlit UI

Full demo interface with:
- Architecture visualization
- Program simulator (Python/C/C++)
- HTML live preview
- FastAPI/Flask launcher with endpoint tester
- Real-time execution traces

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Dev Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

IS_WINDOWS = platform.system() == "Windows"
WORKSPACE = Path("generated_projects/current_project")


# ---------------------------------------------------------------------------
# Custom CSS — Futuristic theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(ellipse at top left, rgba(120, 90, 255, 0.15), transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(0, 220, 255, 0.10), transparent 50%),
            linear-gradient(180deg, #0a0a1a 0%, #050514 100%);
        color: #e0e6f0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #f0f4ff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    h1 {
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 30, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #c4b5fd !important;
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(124, 58, 237, 0.3);
        background: rgba(124, 58, 237, 0.1);
        color: #e0e6f0;
        font-weight: 500;
        transition: all 0.2s ease;
        backdrop-filter: blur(10px);
    }
    .stButton > button:hover {
        background: rgba(124, 58, 237, 0.3);
        border-color: rgba(124, 58, 237, 0.8);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        border: none;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 30px rgba(124, 58, 237, 0.6);
        transform: translateY(-2px);
    }
    .stTextArea textarea, .stTextInput input {
        background: rgba(20, 20, 40, 0.6) !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e6f0 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
    }
    .stCodeBlock, pre {
        background: rgba(10, 10, 25, 0.8) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .streamlit-expanderHeader {
        background: rgba(20, 20, 40, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
    }
    hr {
        border-color: rgba(124, 58, 237, 0.2) !important;
        margin: 2rem 0 !important;
    }
    .neon-card {
        background: rgba(20, 20, 40, 0.5);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    .neon-card:hover {
        border-color: rgba(124, 58, 237, 0.6);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
    }
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .status-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
        border: 1px solid rgba(34, 197, 94, 0.5);
        color: #4ade80;
    }
    .status-failed {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #f87171;
    }
    .status-pending {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fbbf24;
    }
    .arch-path {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #c4b5fd;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stSpinner > div { border-top-color: #7c3aed !important; }
    .proof-item {
        background: rgba(34, 197, 94, 0.08);
        border-left: 3px solid #4ade80;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #d1fae5;
    }
    .terminal-box {
        background: #0a0a1a;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        color: #4ade80;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    .terminal-error {
        background: #1a0a0a;
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        justify-content: flex-start;
        font-size: 0.85rem;
        padding: 8px 12px;
        white-space: normal;
        height: auto;
        min-height: 40px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    ("🐍", "Python — Prime Numbers", "a python program to print prime numbers up to 50"),
    ("⚙️", "C++ — Swap Function", "a cpp program to swap two numbers using a function"),
    ("🔢", "C — Fibonacci", "a C program to compute first 10 fibonacci numbers"),
    ("🌐", "FastAPI — Weather", "a FastAPI weather endpoint, use free open source api"),
    ("🎨", "HTML — Student Form", "create an HTML form for student details with CSS styling, keep separate files"),
]

WEB_FRAMEWORKS = ("fastapi", "flask", "uvicorn", "streamlit", "starlette")
WEB_FILES_HINT = ("main.py", "app.py", "server.py")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "requirement" not in st.session_state:
    st.session_state["requirement"] = ""

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "running_server_proc" not in st.session_state:
    st.session_state["running_server_proc"] = None

if "running_server_port" not in st.session_state:
    st.session_state["running_server_port"] = None

if "running_html_proc" not in st.session_state:
    st.session_state["running_html_proc"] = None

if "running_html_port" not in st.session_state:
    st.session_state["running_html_port"] = None

if "simulator_output" not in st.session_state:
    st.session_state["simulator_output"] = None


# ---------------------------------------------------------------------------
# Subprocess cleanup
# ---------------------------------------------------------------------------

def _cleanup_dead_processes() -> None:
    """Remove dead subprocess handles from session state."""
    for key in ("running_server_proc", "running_html_proc"):
        proc = st.session_state.get(key)
        if proc is not None and proc.poll() is not None:
            st.session_state[key] = None
            port_key = key.replace("_proc", "_port")
            st.session_state[port_key] = None


_cleanup_dead_processes()


def _kill_process(proc: subprocess.Popen) -> None:
    """Cross-platform process-tree kill."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


# ---------------------------------------------------------------------------
# Example click handler
# ---------------------------------------------------------------------------

def _set_example(prompt: str) -> None:
    st.session_state["requirement"] = prompt


# ---------------------------------------------------------------------------
# Project type detection
# ---------------------------------------------------------------------------

def _is_web_app_project(files: list, logs: list) -> bool:
    """Detect FastAPI/Flask/Streamlit projects needing a server launch."""
    if any("WEB EXECUTOR" in entry for entry in logs):
        return True
    req_path = WORKSPACE / "requirements.txt"
    if req_path.exists():
        try:
            content = req_path.read_text(encoding="utf-8", errors="replace").lower()
            if any(fw in content for fw in WEB_FRAMEWORKS):
                return True
        except OSError:
            pass
    for hint_file in WEB_FILES_HINT:
        fpath = WORKSPACE / hint_file
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace").lower()
                if any(f"from {fw}" in content or f"import {fw}" in content
                       for fw in WEB_FRAMEWORKS):
                    return True
            except OSError:
                continue
    return False


def _is_html_project(files: list) -> bool:
    """Detect static HTML projects (without a Python server)."""
    has_html = any(f.lower().endswith(".html") for f in files)
    return has_html


def _is_python_script_project(files: list, logs: list) -> bool:
    """Detect plain Python scripts (not web apps)."""
    if _is_web_app_project(files, logs):
        return False
    py_files = [f for f in files if f.lower().endswith(".py")]
    return len(py_files) > 0


def _is_compiled_project(files: list, logs: list) -> bool:
    """Detect C/C++/Rust/Go projects."""
    if any("COMPILED EXECUTOR" in entry for entry in logs):
        return True
    compiled_exts = (".c", ".cpp", ".cc", ".rs", ".go", ".java")
    return any(f.lower().endswith(compiled_exts) for f in files)


def _find_source_file(files: list, extensions: tuple) -> Optional[str]:
    """Find the first file with one of the given extensions."""
    for f in files:
        if f.lower().endswith(extensions):
            return f
    return None


def _find_html_entry(files: list) -> Optional[str]:
    """Find the main HTML entry point — prefer index.html."""
    html_files = [f for f in files if f.lower().endswith(".html")]
    if not html_files:
        return None
    for f in html_files:
        if "index" in f.lower():
            return f
    return html_files[0]


# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------

def _is_port_in_use(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.3)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def _find_free_port(start: int = 8000, attempts: int = 20) -> int:
    for offset in range(attempts):
        port = start + offset
        if not _is_port_in_use(port):
            return port
    return start


# ---------------------------------------------------------------------------
# Web app server (FastAPI/Flask) launcher
# ---------------------------------------------------------------------------

def _detect_app_module(files: list) -> str:
    for candidate in ("main.py", "app.py", "server.py"):
        if candidate in files:
            module = candidate.replace(".py", "")
            return f"{module}:app"
    return "main:app"


def _launch_uvicorn(app_module: str, port: int) -> Optional[subprocess.Popen]:
    workspace = WORKSPACE.resolve()
    if not workspace.exists():
        return None
    cmd = f"python -m uvicorn {app_module} --port {port} --host 127.0.0.1"
    popen_kwargs = dict(
        shell=True, cwd=str(workspace),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **popen_kwargs)
    except (OSError, ValueError):
        return None


def _wait_for_server_ready(port: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    paths = ("/docs", "/openapi.json", "/")
    while time.monotonic() < deadline:
        for path in paths:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=1.0
                ) as resp:
                    if 200 <= resp.status < 500:
                        return True
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    return True
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# HTML server launcher (python -m http.server)
# ---------------------------------------------------------------------------

def _launch_html_server(port: int) -> Optional[subprocess.Popen]:
    workspace = WORKSPACE.resolve()
    if not workspace.exists():
        return None
    cmd = f"python -m http.server {port} --bind 127.0.0.1"
    popen_kwargs = dict(
        shell=True, cwd=str(workspace),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **popen_kwargs)
    except (OSError, ValueError):
        return None


def _wait_for_html_ready(port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=1.0
            ) as resp:
                return 200 <= resp.status < 500
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# Program simulator (Python / compiled binaries)
# ---------------------------------------------------------------------------

def _run_program(
    command: str,
    stdin_input: str = "",
    timeout: int = 15,
) -> Tuple[int, str, str, float]:
    """
    Run a program with optional stdin input.
    Returns (exit_code, stdout, stderr, elapsed_seconds).
    """
    workspace = WORKSPACE.resolve()
    start = time.monotonic()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            input=stdin_input if stdin_input else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - start
        return result.returncode, result.stdout or "", result.stderr or "", elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return -1, "", f"⏱ Timed out after {timeout} seconds", elapsed
    except Exception as exc:
        elapsed = time.monotonic() - start
        return -2, "", f"💥 Execution error: {exc}", elapsed


def _compile_program(source_file: str) -> Tuple[bool, str, str]:
    """
    Compile a C/C++ source file. Returns (success, binary_name, error).
    """
    workspace = WORKSPACE.resolve()
    binary_name = Path(source_file).stem

    if source_file.lower().endswith((".cpp", ".cc")):
        cmd = f"g++ {source_file} -o {binary_name}"
    elif source_file.lower().endswith(".c"):
        cmd = f"gcc {source_file} -o {binary_name}"
    else:
        return False, "", f"Unsupported file type: {source_file}"

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(workspace),
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return True, binary_name, ""
        return False, "", result.stderr or "Unknown compile error"
    except subprocess.TimeoutExpired:
        return False, "", "Compile timed out after 30s"
    except Exception as exc:
        return False, "", f"Compile error: {exc}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='color: #c4b5fd; margin-bottom: 0;'>🤖 AI Dev Agent</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Autonomous code generation")

    st.markdown("---")

    st.markdown(
    """
    <div class='arch-path' style='text-align: left; font-size: 0.82rem; line-height: 1.55;'>
    <b style='color: #fbbf24;'>Planner</b><br>
    &nbsp;&nbsp;↓<br>
    <b style='color: #06b6d4;'>Orchestrator Agent</b><br>
    &nbsp;&nbsp;↓<br>
    ├─ <span style='color: #4ade80;'>Simple Executor</span><br>
    ├─ <span style='color: #f472b6;'>Compiled Executor</span><br>
    └─ <span style='color: #60a5fa;'>Web Executor</span><br>
    &nbsp;&nbsp;↓<br>
    <b style='color: #a78bfa;'>Execution Result?</b><br>
    &nbsp;&nbsp;├─ <span style='color: #4ade80;'>✓ Success</span> → <b style='color: #4ade80;'>END</b><br>
    &nbsp;&nbsp;└─ <span style='color: #f87171;'>✗ Failure</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<b style='color: #fb923c;'>Error Analyzer</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<b style='color: #fb923c;'>Fix Generator</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<b style='color: #06b6d4;'>Re-execute</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<span style='color: #fbbf24;'>Retries &lt; 3?</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ <span style='color: #4ade80;'>Yes</span> → retry loop<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <span style='color: #f87171;'>No</span> → <b style='color: #fbbf24;'>Replan</b>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.markdown("### 🛠️ Host Tools")

    try:
        from src.tools.library_manager import detect_all

        if "toolchain_cache" not in st.session_state:
            with st.spinner("Detecting toolchains..."):
                st.session_state["toolchain_cache"] = detect_all()

        toolchains = st.session_state["toolchain_cache"]

        # Group by category
        categories: Dict[str, List] = {}
        for name, status in toolchains.items():
            categories.setdefault(status["category"], []).append(status)

        category_order = ["python", "node", "compiled", "vcs", "container"]
        category_labels = {
            "python": "🐍 Python",
            "node": "📦 Node.js",
            "compiled": "⚙️ Compiled",
            "vcs": "🔧 VCS",
            "container": "🐳 Containers",
        }

        for cat in category_order:
            if cat not in categories:
                continue
            with st.expander(category_labels.get(cat, cat), expanded=False):
                for status in categories[cat]:
                    icon = "✓" if status["available"] else "✗"
                    color = "#4ade80" if status["available"] else "#f87171"
                    label = status["label"]
                    extra = (
                        f"<span style='color: #94a3b8; font-size: 0.75rem;'>"
                        f" — {status['version'][:30]}</span>"
                        if status["version"] else ""
                    )
                    st.markdown(
                        f"<div style='font-family: monospace; font-size: 0.85rem;'>"
                        f"<span style='color: {color};'>{icon}</span> "
                        f"<b>{label}</b>{extra}</div>",
                        unsafe_allow_html=True,
                    )

        # Refresh button
        if st.button("🔄 Re-scan", use_container_width=True, key="rescan_tools"):
            from src.tools.library_manager import reset_cache
            reset_cache()
            st.session_state.pop("toolchain_cache", None)
            st.rerun()

    except Exception as exc:
        st.caption(f"⚠ Toolchain detection failed: {exc}")

    st.markdown("---")

    st.markdown("### 📌 Examples")
    st.caption("Click to load into input")

    for emoji, label, prompt in EXAMPLE_PROMPTS:
        st.button(
            f"{emoji}  {label}",
            key=f"ex_{label}",
            use_container_width=True,
            on_click=_set_example,
            args=(prompt,),
        )

    st.markdown("---")

    st.markdown("### ⚙️ Options")
    show_full_trace = st.checkbox("Show full trace", value=False)
    show_files = st.checkbox("Show file contents", value=True)

    st.markdown("---")
    st.caption("🛡️ 41 tests passing")
    st.caption("🚀 LangGraph-powered")



# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='margin-bottom: 1rem;'>
        <h1 style='margin: 0; font-size: 3rem;'>AI Dev Agent</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>
            Type a requirement. Watch the agent classify, route, generate, and verify.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='display: flex; gap: 12px; margin-bottom: 2rem;'>
        <span class='status-pill status-success'>● Online</span>
        <span style='color: #94a3b8; font-size: 0.85rem; align-self: center;'>
            Models loaded · LangGraph compiled · Workspace ready
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

st.markdown("### 💬 Your Requirement")

requirement = st.text_area(
    "Requirement",
    key="requirement",
    placeholder="e.g. a python program to find the largest number in a list...",
    height=120,
    label_visibility="collapsed",
)

col_run, col_clear = st.columns([5, 1])

with col_run:
    run_clicked = st.button(
        "🚀  Run Agent",
        type="primary",
        use_container_width=True,
        disabled=not requirement.strip(),
    )

with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state["requirement"] = ""
        st.session_state["last_result"] = None
        st.session_state["simulator_output"] = None
        if st.session_state.get("running_server_proc"):
            _kill_process(st.session_state["running_server_proc"])
            st.session_state["running_server_proc"] = None
        if st.session_state.get("running_html_proc"):
            _kill_process(st.session_state["running_html_proc"])
            st.session_state["running_html_proc"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Result rendering helpers
# ---------------------------------------------------------------------------

def _render_status_banner(status: str, error: Optional[str], elapsed: float) -> None:
    pill_class = {"success": "status-success", "failed": "status-failed"}.get(
        status, "status-pending"
    )
    label = {"success": "✓ Success", "failed": "✗ Failed"}.get(
        status, "⚠ Incomplete"
    )
    msg = {
        "success": "Agent completed all steps",
        "failed": error or "Unknown error",
    }.get(status, error or "Status unclear")
    border_color = {
        "success": "rgba(34, 197, 94, 0.5)",
        "failed": "rgba(239, 68, 68, 0.5)",
    }.get(status, "rgba(245, 158, 11, 0.5)")

    st.markdown(
        f"""
        <div class='neon-card' style='border-color: {border_color};'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <span class='status-pill {pill_class}'>{label}</span>
                    <span style='color: #e0e6f0; margin-left: 12px;'>{msg}</span>
                </div>
                <span style='color: #94a3b8; font-family: monospace;'>⏱ {elapsed:.1f}s</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(plan: list, files: list, logs: list) -> None:
    verifications = sum(
        1 for l in logs
        if "verified:" in l or "Verified live" in l or "Endpoint responded" in l
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Plan Steps", len(plan))
    col2.metric("Files Created", len(files))
    col3.metric("Log Entries", len(logs))
    col4.metric("Verifications", verifications)


def _render_architecture_trace(logs: list) -> None:
    markers = {
        "PLANNER NODE EXECUTING": ("🧭", "Planner", "#fbbf24"),
        "ROUTER NODE EXECUTING": ("🚦", "Router", "#06b6d4"),
        "SIMPLE EXECUTOR": ("🐍", "Simple Executor", "#4ade80"),
        "COMPILED EXECUTOR": ("⚙️", "Compiled Executor", "#f472b6"),
        "WEB EXECUTOR": ("🌐", "Web Executor", "#60a5fa"),
    }
    sequence = []
    for entry in logs:
        for marker, (emoji, label, color) in markers.items():
            if marker in entry:
                if not sequence or sequence[-1][1] != label:
                    sequence.append((emoji, label, color))
    if not sequence:
        return

    st.markdown("### 🏗️ Execution Path")
    path_html = "<div class='arch-path'>"
    for i, (emoji, label, color) in enumerate(sequence):
        if i > 0:
            path_html += " <span style='color: #64748b; margin: 0 8px;'>→</span> "
        path_html += f"<span style='color: {color};'>{emoji} {label}</span>"
    path_html += "</div>"
    st.markdown(path_html, unsafe_allow_html=True)


def _render_plan(plan: list) -> None:
    st.markdown("### 📋 Generated Plan")
    if not plan:
        st.info("No plan generated.")
        return
    plan_html = "<div class='neon-card'>"
    for i, step in enumerate(plan, 1):
        plan_html += (
            f"<div style='display: flex; gap: 12px; margin: 8px 0;'>"
            f"<span style='color: #7c3aed; font-weight: 700; min-width: 24px;'>{i}.</span>"
            f"<span style='color: #e0e6f0;'>{step}</span></div>"
        )
    plan_html += "</div>"
    st.markdown(plan_html, unsafe_allow_html=True)


def _render_verification_proof(logs: list) -> None:
    markers = ("Verified live:", "Endpoint responded:", "[verified:", "exit=0")
    proofs = []
    seen = set()
    for entry in logs:
        if any(marker in entry for marker in markers):
            key = entry.strip()
            if key not in seen:
                seen.add(key)
                proofs.append(entry)
    if not proofs:
        return
    st.markdown("### ✓ Verification Proof")
    for proof in proofs:
        text = proof.strip()
        if len(text) > 200:
            text = text[:200] + "..."
        st.markdown(f"<div class='proof-item'>{text}</div>", unsafe_allow_html=True)


def _render_files(files: list, show_contents: bool) -> None:
    st.markdown("### 📁 Generated Files")
    if not files:
        st.info("No files were created.")
        return
    for file_rel_path in files:
        full_path = WORKSPACE / file_rel_path
        with st.expander(f"📄  `{file_rel_path}`", expanded=False):
            if not full_path.exists():
                st.warning("File path tracked but missing on disk.")
                continue
            if show_contents:
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    suffix = full_path.suffix.lstrip(".")
                    lang_map = {
                        "py": "python", "js": "javascript", "ts": "typescript",
                        "cpp": "cpp", "c": "c", "rs": "rust", "go": "go",
                        "java": "java", "html": "html", "css": "css",
                        "json": "json", "yaml": "yaml", "yml": "yaml",
                        "md": "markdown", "sh": "bash",
                    }
                    lang = lang_map.get(suffix.lower(), "text")
                    st.code(content, language=lang)
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")


def _render_warnings(logs: list) -> None:
    markers = ("ERROR RECOVERY", "✗", "CRITICAL ERROR", "Validation failed")
    issues = []
    seen = set()
    for entry in logs:
        if any(m in entry for m in markers):
            key = entry.strip()
            if key not in seen:
                seen.add(key)
                issues.append(entry)
    if not issues:
        return
    st.markdown("### ⚠️ Warnings & Recovery Events")
    for issue in issues[:8]:
        st.warning(f"⚠ {issue.strip()[:200]}")
    if len(issues) > 8:
        st.caption(f"... and {len(issues) - 8} more")


def _render_full_trace(logs: list) -> None:
    st.markdown("### 📜 Full Execution Log")
    if not logs:
        st.info("No log entries.")
        return
    log_text = "\n".join(f"{i:>3}. {entry}" for i, entry in enumerate(logs, 1))
    st.code(log_text, language="text")


# ---------------------------------------------------------------------------
# Interactive runners (NEW)
# ---------------------------------------------------------------------------

def _render_web_app_launcher(files: list) -> None:
    """FastAPI/Flask server launcher with endpoint tester."""
    st.markdown("---")
    st.markdown("### 🌐 FastAPI / Flask Launcher")

    proc = st.session_state.get("running_server_proc")
    port = st.session_state.get("running_server_port", 8000)
    is_running = proc is not None and proc.poll() is None

    if is_running:
        base_url = f"http://localhost:{port}"
        st.markdown(
            f"""
            <div class='neon-card' style='border-color: rgba(34, 197, 94, 0.5);'>
                <span class='status-pill status-success'>● Running</span>
                <span style='color: #d1fae5; margin-left: 12px; font-family: monospace;'>
                    {base_url}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.link_button("📘 Swagger", f"{base_url}/docs", use_container_width=True)
        with col_b:
            st.link_button("🏠 Root", f"{base_url}/", use_container_width=True)
        with col_c:
            st.link_button("📋 OpenAPI", f"{base_url}/openapi.json",
                           use_container_width=True)
        with col_d:
            if st.button("⏹ Stop", use_container_width=True, key="stop_uvicorn"):
                _kill_process(proc)
                st.session_state["running_server_proc"] = None
                st.session_state["running_server_port"] = None
                st.success("Server stopped.")
                time.sleep(0.4)
                st.rerun()

        st.markdown("**🔍 Test Any Endpoint**")
        col_path, col_test = st.columns([4, 1])
        with col_path:
            path = st.text_input(
                "path", value="/docs",
                key="endpoint_path",
                label_visibility="collapsed",
                placeholder="/weather?city=London",
            )
        with col_test:
            test_clicked = st.button("Test", use_container_width=True,
                                     key="test_endpoint")

        if test_clicked and path.strip():
            url = f"{base_url}{path.strip()}"
            try:
                with urllib.request.urlopen(url, timeout=5.0) as resp:
                    body = resp.read(4096).decode("utf-8", errors="replace")
                    st.success(f"✅ HTTP {resp.status} from `{url}`")
                    try:
                        parsed = json.loads(body)
                        st.code(json.dumps(parsed, indent=2)[:2000], language="json")
                    except (ValueError, json.JSONDecodeError):
                        st.code(body[:2000], language="text")
            except urllib.error.HTTPError as exc:
                st.error(f"❌ HTTP {exc.code} — {exc.reason}")
                try:
                    err_body = exc.read(2000).decode("utf-8", errors="replace")
                    st.code(err_body, language="text")
                except Exception:
                    pass
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                st.error(f"❌ Network error: {exc}")
        return

    # Not running — show launch button
    app_module = _detect_app_module(files)
    st.markdown(
        f"""
        <div class='neon-card'>
            <p style='color: #94a3b8; margin: 0 0 8px 0;'>
                Launch the generated web app to test endpoints in your browser.
            </p>
            <p style='color: #c4b5fd; font-family: monospace; margin: 0;'>
                python -m uvicorn {app_module}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚀 Launch Server", type="primary",
                 use_container_width=True, key="launch_uvicorn"):
        with st.spinner("Starting server..."):
            free_port = _find_free_port(8000)
            new_proc = _launch_uvicorn(app_module, free_port)
            if new_proc is None:
                st.error("Failed to launch. Is uvicorn installed?")
                return
            if not _wait_for_server_ready(free_port, timeout=10.0):
                _kill_process(new_proc)
                try:
                    _, stderr = new_proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    stderr = ""
                st.error(f"Server didn't respond within 10 seconds (port {free_port})")
                if stderr:
                    with st.expander("Show stderr"):
                        st.code(stderr[:2000], language="text")
                return
            st.session_state["running_server_proc"] = new_proc
            st.session_state["running_server_port"] = free_port
            st.success(f"✅ Server running at http://localhost:{free_port}")
            time.sleep(0.4)
            st.rerun()


def _render_html_preview(files: list) -> None:
    """Live HTML server with embedded preview."""
    st.markdown("---")
    st.markdown("### 🎨 HTML Live Preview")

    proc = st.session_state.get("running_html_proc")
    port = st.session_state.get("running_html_port", 8080)
    is_running = proc is not None and proc.poll() is None

    entry_file = _find_html_entry(files) or "index.html"

    if is_running:
        url = f"http://localhost:{port}/{entry_file}"
        st.markdown(
            f"""
            <div class='neon-card' style='border-color: rgba(34, 197, 94, 0.5);'>
                <span class='status-pill status-success'>● Live</span>
                <span style='color: #d1fae5; margin-left: 12px; font-family: monospace;'>
                    {url}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_open, col_stop = st.columns([3, 1])
        with col_open:
            st.link_button("🌐 Open in new tab", url, use_container_width=True)
        with col_stop:
            if st.button("⏹ Stop", use_container_width=True, key="stop_html"):
                _kill_process(proc)
                st.session_state["running_html_proc"] = None
                st.session_state["running_html_port"] = None
                st.success("Server stopped.")
                time.sleep(0.4)
                st.rerun()

        # Embedded iframe preview
        st.markdown("**📺 Live Preview**")
        st.components.v1.iframe(url, height=600, scrolling=True)
        return

    st.markdown(
        f"""
        <div class='neon-card'>
            <p style='color: #94a3b8; margin: 0 0 8px 0;'>
                Start a local HTTP server to preview the HTML/CSS/JS files.
            </p>
            <p style='color: #c4b5fd; font-family: monospace; margin: 0;'>
                python -m http.server  →  serves <b>{entry_file}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start Live Server", type="primary",
                 use_container_width=True, key="launch_html"):
        with st.spinner("Starting HTML server..."):
            free_port = _find_free_port(8080)
            new_proc = _launch_html_server(free_port)
            if new_proc is None:
                st.error("Failed to launch HTML server.")
                return
            if not _wait_for_html_ready(free_port, timeout=5.0):
                _kill_process(new_proc)
                st.error("HTML server didn't respond. Check the files in workspace.")
                return
            st.session_state["running_html_proc"] = new_proc
            st.session_state["running_html_port"] = free_port
            st.success(f"✅ Live server at http://localhost:{free_port}")
            time.sleep(0.4)
            st.rerun()


def _render_program_simulator(files: list, project_type: str) -> None:
    """Interactive simulator for Python scripts and compiled programs."""
    st.markdown("---")

    if project_type == "python":
        st.markdown("### 🐍 Python Simulator")
        py_file = _find_source_file(files, (".py",))
        if not py_file:
            st.info("No Python file found to run.")
            return
        st.markdown(
            f"""
            <div class='neon-card'>
                <p style='color: #94a3b8; margin: 0 0 8px 0;'>
                    Run the generated Python script with optional stdin input.
                </p>
                <p style='color: #c4b5fd; font-family: monospace; margin: 0;'>
                    python {py_file}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        command = f"python {py_file}"

    elif project_type == "compiled":
        st.markdown("### ⚙️ Compiled Program Simulator")
        source_file = _find_source_file(files, (".cpp", ".cc", ".c"))
        if not source_file:
            st.info("No C/C++ source file found.")
            return
        binary_name = Path(source_file).stem
        binary_file = binary_name + (".exe" if IS_WINDOWS else "")
        binary_exists = (WORKSPACE / binary_file).exists() or (WORKSPACE / binary_name).exists()

        st.markdown(
            f"""
            <div class='neon-card'>
                <p style='color: #94a3b8; margin: 0 0 8px 0;'>
                    {"Run the compiled binary" if binary_exists else "Compile and run the source"} with optional stdin input.
                </p>
                <p style='color: #c4b5fd; font-family: monospace; margin: 0;'>
                    {f"./{binary_name}" if binary_exists else f"g++/gcc {source_file} -o {binary_name} && ./{binary_name}"}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not binary_exists:
            if st.button("🔨 Compile First", use_container_width=True, key="compile_btn"):
                with st.spinner(f"Compiling {source_file}..."):
                    ok, name, err = _compile_program(source_file)
                    if ok:
                        st.success(f"✅ Compiled successfully → `{name}`")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Compilation failed")
                        st.code(err, language="text")
            return

        run_cmd = f"{binary_name}.exe" if IS_WINDOWS else f"./{binary_name}"
        command = run_cmd

    else:
        return

    # ─── Stdin input + Run button ─────────────────────────────────────
    st.markdown("**📥 Optional stdin input** *(leave empty if program doesn't need input)*")
    stdin_text = st.text_area(
        "stdin",
        key=f"stdin_{project_type}",
        height=80,
        label_visibility="collapsed",
        placeholder="e.g. 5\n10\n15  (one value per line if program reads multiple inputs)",
    )

    col_run, col_clear = st.columns([4, 1])
    with col_run:
        run_sim = st.button(
            "▶️ Run Program",
            type="primary",
            use_container_width=True,
            key=f"run_sim_{project_type}",
        )
    with col_clear:
        if st.button("Clear Output", use_container_width=True, key=f"clear_sim_{project_type}"):
            st.session_state["simulator_output"] = None
            st.rerun()

    if run_sim:
        with st.spinner(f"Running {command}..."):
            rc, stdout, stderr, elapsed = _run_program(
                command, stdin_input=stdin_text, timeout=15,
            )
            st.session_state["simulator_output"] = {
                "command": command,
                "exit_code": rc,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed": elapsed,
            }

    # ─── Render output if exists ─────────────────────────────────────
    output = st.session_state.get("simulator_output")
    if output:
        st.markdown("**📤 Output**")
        success = output["exit_code"] == 0

        # Status header
        if success:
            st.markdown(
                f"""
                <div class='neon-card' style='border-color: rgba(34, 197, 94, 0.5); padding: 0.8rem 1.2rem;'>
                    <span class='status-pill status-success'>● Exit 0</span>
                    <span style='color: #d1fae5; margin-left: 12px; font-family: monospace;'>
                        ⏱ {output['elapsed']:.2f}s
                    </span>
                    <span style='color: #94a3b8; margin-left: 12px;'>
                        {output['command']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='neon-card' style='border-color: rgba(239, 68, 68, 0.5); padding: 0.8rem 1.2rem;'>
                    <span class='status-pill status-failed'>● Exit {output['exit_code']}</span>
                    <span style='color: #fecaca; margin-left: 12px; font-family: monospace;'>
                        ⏱ {output['elapsed']:.2f}s
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # stdout
        if output["stdout"]:
            st.markdown("**stdout**")
            st.markdown(
                f"<div class='terminal-box'>{output['stdout'][:5000]}</div>",
                unsafe_allow_html=True,
            )

        # stderr
        if output["stderr"]:
            st.markdown("**stderr**")
            st.markdown(
                f"<div class='terminal-box terminal-error'>{output['stderr'][:5000]}</div>",
                unsafe_allow_html=True,
            )

        if not output["stdout"] and not output["stderr"]:
            st.info("Program produced no output.")


def _render_runners(files: list, logs: list) -> None:
    """Decide which runner UI to show based on project type."""
    if _is_web_app_project(files, logs):
        _render_web_app_launcher(files)
    elif _is_html_project(files):
        _render_html_preview(files)
    elif _is_compiled_project(files, logs):
        _render_program_simulator(files, "compiled")
    elif _is_python_script_project(files, logs):
        _render_program_simulator(files, "python")


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

if run_clicked and requirement.strip():
    st.markdown("---")
    st.markdown(
        f"<div style='color: #94a3b8; font-style: italic; margin-bottom: 1rem;'>"
        f"🎯 Processing: <span style='color: #c4b5fd;'>{requirement}</span></div>",
        unsafe_allow_html=True,
    )

    initial_state = AgentState(
        requirement=requirement.strip(),
        plan=[], files=[], logs=[], current_step=0,
        is_complete=False, last_error=None, retry_count=0,
        plan_feedback=None, user_feedback=None,
    )

    with st.spinner("🤖 Agent is thinking, generating, and verifying..."):
        try:
            from src.agent.graph import graph
            start_time = time.monotonic()
            result = graph.invoke(initial_state)
            elapsed = time.monotonic() - start_time
            st.session_state["last_result"] = (result, elapsed)
            st.session_state["simulator_output"] = None  # reset old output
        except Exception as exc:
            st.error(f"💥 Fatal error: {exc}")
            with st.expander("Show traceback"):
                import traceback
                st.code(traceback.format_exc(), language="text")
            st.stop()


# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------

if st.session_state.get("last_result"):
    result, elapsed = st.session_state["last_result"]
    plan = result.get("plan", []) or []
    files = result.get("files", []) or []
    logs = result.get("logs", []) or []
    is_complete = result.get("is_complete", False)
    last_error = result.get("last_error")
    final_status = result.get("final_status", "unknown")

    if not final_status or final_status == "unknown":
        if is_complete and not last_error:
            final_status = "success"
        elif last_error:
            final_status = "failed"
        else:
            final_status = "incomplete"

    st.markdown("---")
    _render_status_banner(final_status, last_error, elapsed)
    st.markdown("")
    _render_metrics(plan, files, logs)
    st.markdown("")
    _render_architecture_trace(logs)
    st.markdown("")
    _render_plan(plan)
    st.markdown("")
    _render_verification_proof(logs)
    st.markdown("")
    _render_files(files, show_files)
    _render_warnings(logs)

    # NEW: Interactive runners based on project type
    _render_runners(files, logs)

    if show_full_trace:
        st.markdown("")
        _render_full_trace(logs)

elif not run_clicked:
    # Welcome state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='neon-card' style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🐍</div>
                <h3 style='margin: 0; color: #4ade80;'>Simple</h3>
                <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>
                    Python scripts<br>HTML / CSS<br>
                    <span style='color: #64748b;'>+ simulator & live preview</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='neon-card' style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>⚙️</div>
                <h3 style='margin: 0; color: #f472b6;'>Compiled</h3>
                <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>
                    C, C++, Rust, Go<br>
                    Auto-compile + run<br>
                    <span style='color: #64748b;'>with stdin support</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='neon-card' style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🌐</div>
                <h3 style='margin: 0; color: #60a5fa;'>Web</h3>
                <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>
                    FastAPI, Flask<br>
                    Launch + test endpoints<br>
                    <span style='color: #64748b;'>inline JSON viewer</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align: center; margin-top: 4rem; color: #64748b; font-size: 0.85rem;'>
        <span style='color: #c4b5fd;'>●</span> LangGraph workflow
        &nbsp;·&nbsp;
        <span style='color: #06b6d4;'>●</span> Classification routing
        &nbsp;·&nbsp;
        <span style='color: #4ade80;'>●</span> Real HTTP verification
        &nbsp;·&nbsp;
        <span style='color: #fbbf24;'>●</span> 41 tests passing
    </div>
    """,
    unsafe_allow_html=True,
)