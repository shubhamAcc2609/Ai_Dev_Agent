"""
AI Dev Agent — Minimal Streamlit UI

Clean, professional interface with:
- Dark / Light theme toggle
- Project-type-aware launcher (Python, Node, FastAPI, HTML, C/C++)
- Real-time execution traces and recovery loop visualization

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
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

IS_WINDOWS = platform.system() == "Windows"
WORKSPACE = Path("generated_projects/current_project")


# ---------------------------------------------------------------------------
# Theme management
# ---------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"


def _theme_css(mode: str) -> str:
    """Return CSS for either dark or light theme."""
    if mode == "light":
        return """
        <style>
            .stApp {
                background: #fafafa;
                color: #1a1a1a;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
                max-width: 1100px;
            }
            h1, h2, h3 { color: #0f172a !important; font-weight: 600; }
            h1 { font-size: 2.25rem !important; letter-spacing: -0.02em; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.1rem !important; }
            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e5e7eb;
            }
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 { color: #1f2937 !important; }
            .stButton > button {
                border-radius: 8px;
                border: 1px solid #d1d5db;
                background: #ffffff;
                color: #1f2937;
                font-weight: 500;
            }
            .stButton > button:hover {
                background: #f3f4f6;
                border-color: #9ca3af;
            }
            .stButton > button[kind="primary"] {
                background: #4f46e5;
                border: 1px solid #4f46e5;
                color: white;
            }
            .stButton > button[kind="primary"]:hover {
                background: #4338ca;
                border-color: #4338ca;
            }
            .stTextArea textarea, .stTextInput input {
                background: #ffffff !important;
                border: 1px solid #d1d5db !important;
                border-radius: 8px !important;
                color: #1a1a1a !important;
                font-family: 'JetBrains Mono', 'Consolas', monospace !important;
            }
            .stTextArea textarea:focus, .stTextInput input:focus {
                border-color: #4f46e5 !important;
                box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
            }
            .stCodeBlock, pre {
                background: #f9fafb !important;
                border: 1px solid #e5e7eb !important;
                border-radius: 8px !important;
            }
            [data-testid="stMetricValue"] {
                color: #4f46e5 !important;
                font-weight: 600 !important;
            }
            [data-testid="stMetricLabel"] {
                color: #6b7280 !important;
                font-size: 0.7rem !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            hr { border-color: #e5e7eb !important; margin: 1.5rem 0 !important; }
            .card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 1.25rem;
                margin: 0.75rem 0;
            }
            .status-pill {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .status-success { background: #d1fae5; color: #047857; }
            .status-failed { background: #fee2e2; color: #b91c1c; }
            .status-pending { background: #fef3c7; color: #b45309; }
            .arch-path {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.75rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                color: #374151;
            }
            .terminal-box {
                background: #f9fafb;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 0.75rem;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 0.8rem;
                color: #1f2937;
                white-space: pre-wrap;
                max-height: 360px;
                overflow-y: auto;
            }
            .terminal-error { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
            #MainMenu, footer, .stDeployButton { display: none; }
        </style>
        """
    # Dark mode (default)
    return """
    <style>
        .stApp {
            background: #0b0d12;
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1100px;
        }
        h1, h2, h3 { color: #f3f4f6 !important; font-weight: 600; }
        h1 { font-size: 2.25rem !important; letter-spacing: -0.02em; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.1rem !important; }
        section[data-testid="stSidebar"] {
            background: #0f1218;
            border-right: 1px solid #1f2937;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: #d1d5db !important; }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #2d3340;
            background: #161a22;
            color: #e5e7eb;
            font-weight: 500;
        }
        .stButton > button:hover {
            background: #1f2530;
            border-color: #3b4150;
        }
        .stButton > button[kind="primary"] {
            background: #6366f1;
            border: 1px solid #6366f1;
            color: white;
        }
        .stButton > button[kind="primary"]:hover {
            background: #4f46e5;
            border-color: #4f46e5;
        }
        .stTextArea textarea, .stTextInput input {
            background: #0f1218 !important;
            border: 1px solid #2d3340 !important;
            border-radius: 8px !important;
            color: #e5e7eb !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        }
        .stCodeBlock, pre {
            background: #0a0c11 !important;
            border: 1px solid #1f2937 !important;
            border-radius: 8px !important;
        }
        [data-testid="stMetricValue"] {
            color: #818cf8 !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        hr { border-color: #1f2937 !important; margin: 1.5rem 0 !important; }
        .card {
            background: #11141b;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 1.25rem;
            margin: 0.75rem 0;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .status-success { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .status-failed { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .status-pending { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .arch-path {
            background: #0f1218;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        .terminal-box {
            background: #0a0c11;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 0.75rem;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 0.8rem;
            color: #86efac;
            white-space: pre-wrap;
            max-height: 360px;
            overflow-y: auto;
        }
        .terminal-error { background: #1a0a0a; border-color: #3f1d1d; color: #fca5a5; }
        #MainMenu, footer, .stDeployButton { display: none; }
    </style>
    """


st.markdown(_theme_css(st.session_state["theme"]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    ("Python", "Prime Numbers", "a python program to print prime numbers up to 50"),
    ("C++", "Swap Function", "a cpp program to swap two numbers using a function"),
    ("C", "Fibonacci", "a C program to compute first 10 fibonacci numbers"),
    ("Node", "Express Hello", "an Express.js server on port 8080 with a /hello endpoint"),
    ("FastAPI", "Weather", "a FastAPI weather endpoint, use free open source api"),
    ("HTML", "Student Form", "create an HTML form for student details with CSS styling, keep separate files"),
]

WEB_FRAMEWORKS = ("fastapi", "flask", "uvicorn", "streamlit", "starlette")
WEB_FILES_HINT = ("main.py", "app.py", "server.py")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

for key, default in [
    ("requirement", ""),
    ("last_result", None),
    ("running_server_proc", None),
    ("running_server_port", None),
    ("running_html_proc", None),
    ("running_html_port", None),
    ("running_node_proc", None),
    ("running_node_port", None),
    ("simulator_output", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _cleanup_dead_processes() -> None:
    for key in ("running_server_proc", "running_html_proc", "running_node_proc"):
        proc = st.session_state.get(key)
        if proc is not None and proc.poll() is not None:
            st.session_state[key] = None
            port_key = key.replace("_proc", "_port")
            st.session_state[port_key] = None


_cleanup_dead_processes()


def _kill_process(proc: subprocess.Popen) -> None:
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


def _set_example(prompt: str) -> None:
    st.session_state["requirement"] = prompt


# ---------------------------------------------------------------------------
# Project type detection
# ---------------------------------------------------------------------------

def _is_node_project(files: list) -> bool:
    return any(f.lower() == "package.json" for f in files)


def _is_web_app_project(files: list, logs: list) -> bool:
    if _is_node_project(files):
        return False
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
    return any(f.lower().endswith(".html") for f in files)


def _is_python_script_project(files: list, logs: list) -> bool:
    if _is_web_app_project(files, logs) or _is_node_project(files):
        return False
    return any(f.lower().endswith(".py") for f in files)


def _is_compiled_project(files: list, logs: list) -> bool:
    if any("COMPILED EXECUTOR" in entry for entry in logs):
        return True
    compiled_exts = (".c", ".cpp", ".cc", ".rs", ".go", ".java")
    return any(f.lower().endswith(compiled_exts) for f in files)


def _find_source_file(files: list, extensions: tuple) -> Optional[str]:
    for f in files:
        if f.lower().endswith(extensions):
            return f
    return None


def _find_html_entry(files: list) -> Optional[str]:
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
# Server launchers
# ---------------------------------------------------------------------------

def _detect_app_module(files: list) -> str:
    for candidate in ("main.py", "app.py", "server.py"):
        if candidate in files:
            return f"{candidate.replace('.py', '')}:app"
    return "main:app"


def _detect_node_entry() -> Tuple[str, str]:
    """Returns (run_command, entry_file). Reads package.json for start script."""
    pkg_path = WORKSPACE / "package.json"
    if pkg_path.exists():
        try:
            with open(pkg_path) as f:
                pkg = json.load(f)
            start = pkg.get("scripts", {}).get("start")
            if start:
                return "npm start", pkg.get("main", "index.js")
            main = pkg.get("main", "index.js")
            return f"node {main}", main
        except Exception:
            pass
    return "node index.js", "index.js"


def _launch_uvicorn(app_module: str, port: int) -> Optional[subprocess.Popen]:
    workspace = WORKSPACE.resolve()
    if not workspace.exists():
        return None
    cmd = f"python -m uvicorn {app_module} --port {port} --host 127.0.0.1"
    return _spawn(cmd, workspace)


def _launch_html_server(port: int) -> Optional[subprocess.Popen]:
    workspace = WORKSPACE.resolve()
    if not workspace.exists():
        return None
    cmd = f"python -m http.server {port} --bind 127.0.0.1"
    return _spawn(cmd, workspace)


def _launch_node_server(run_cmd: str) -> Optional[subprocess.Popen]:
    workspace = WORKSPACE.resolve()
    if not workspace.exists():
        return None
    return _spawn(run_cmd, workspace)


def _spawn(cmd: str, cwd: Path) -> Optional[subprocess.Popen]:
    popen_kwargs = dict(
        shell=True, cwd=str(cwd),
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


def _wait_for_server_ready(port: int, timeout: float = 8.0,
                            paths: tuple = ("/docs", "/", "/health")) -> bool:
    deadline = time.monotonic() + timeout
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
# Program runner
# ---------------------------------------------------------------------------

def _run_program(command: str, stdin_input: str = "",
                 timeout: int = 15) -> Tuple[int, str, str, float]:
    workspace = WORKSPACE.resolve()
    start = time.monotonic()
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(workspace),
            input=stdin_input if stdin_input else None,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        elapsed = time.monotonic() - start
        return result.returncode, result.stdout or "", result.stderr or "", elapsed
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout} seconds", time.monotonic() - start
    except Exception as exc:
        return -2, "", f"Execution error: {exc}", time.monotonic() - start


def _compile_program(source_file: str) -> Tuple[bool, str, str]:
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
    # Header with theme toggle
    col_title, col_theme = st.columns([3, 1])
    with col_title:
        st.markdown("### AI Dev Agent")
    with col_theme:
        if st.button("◐", help="Toggle theme", key="theme_toggle"):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()

    st.caption("Autonomous code generation")
    st.markdown("---")

    # Architecture flow
    st.markdown("##### Workflow")
    st.markdown(
        """
        <div class='arch-path' style='line-height: 1.7;'>
        <b>Planner</b><br>
        &nbsp;&nbsp;↓<br>
        <b>Orchestrator</b><br>
        &nbsp;&nbsp;↓<br>
        Simple · Compiled · Web<br>
        &nbsp;&nbsp;↓<br>
        <b>Result?</b><br>
        &nbsp;&nbsp;├─ Success → <b>END</b><br>
        &nbsp;&nbsp;└─ Failure<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Error Analyzer<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Fix Generator<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Retry (max 3)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Replan if exhausted
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Host tools
    st.markdown("##### Host Tools")
    try:
        from src.tools.library_manager import detect_all

        if "toolchain_cache" not in st.session_state:
            with st.spinner("Detecting..."):
                st.session_state["toolchain_cache"] = detect_all()

        toolchains = st.session_state["toolchain_cache"]
        categories: Dict[str, List] = {}
        for name, status in toolchains.items():
            categories.setdefault(status["category"], []).append(status)

        category_order = ["python", "node", "compiled", "vcs", "container"]
        category_labels = {
            "python": "Python", "node": "Node.js", "compiled": "Compiled",
            "vcs": "Git", "container": "Containers",
        }

        for cat in category_order:
            if cat not in categories:
                continue
            with st.expander(category_labels.get(cat, cat), expanded=False):
                for status in categories[cat]:
                    icon = "✓" if status["available"] else "✗"
                    color = "#10b981" if status["available"] else "#ef4444"
                    label = status["label"]
                    extra = (
                        f"<span style='color: #6b7280; font-size: 0.75rem;'>"
                        f" — {status['version'][:30]}</span>"
                        if status["version"] else ""
                    )
                    st.markdown(
                        f"<div style='font-family: monospace; font-size: 0.82rem;'>"
                        f"<span style='color: {color};'>{icon}</span> "
                        f"<b>{label}</b>{extra}</div>",
                        unsafe_allow_html=True,
                    )

        if st.button("Re-scan", use_container_width=True, key="rescan_tools"):
            from src.tools.library_manager import reset_cache
            reset_cache()
            st.session_state.pop("toolchain_cache", None)
            st.rerun()
    except Exception as exc:
        st.caption(f"Detection failed: {exc}")

    st.markdown("---")

    # Examples
    st.markdown("##### Examples")
    for tag, label, prompt in EXAMPLE_PROMPTS:
        st.button(
            f"{tag} · {label}",
            key=f"ex_{label}",
            use_container_width=True,
            on_click=_set_example,
            args=(prompt,),
        )

    st.markdown("---")

    # Options
    st.markdown("##### Options")
    show_full_trace = st.checkbox("Show full trace", value=False)
    show_files = st.checkbox("Show file contents", value=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# AI Dev Agent")
st.caption("Type a requirement. Watch the agent classify, route, generate, and verify.")

# Status pill
st.markdown(
    """
    <div style='margin-bottom: 1.5rem;'>
        <span class='status-pill status-success'>● Online</span>
        <span style='color: #6b7280; font-size: 0.8rem; margin-left: 8px;'>
            Models loaded · Workspace ready
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

st.markdown("##### Requirement")

requirement = st.text_area(
    "Requirement",
    key="requirement",
    placeholder="e.g. a python program to find the largest number in a list",
    height=110,
    label_visibility="collapsed",
)

col_run, col_clear = st.columns([5, 1])
with col_run:
    run_clicked = st.button(
        "Run Agent",
        type="primary",
        use_container_width=True,
        disabled=not requirement.strip(),
    )
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state["requirement"] = ""
        st.session_state["last_result"] = None
        st.session_state["simulator_output"] = None
        for key in ("running_server_proc", "running_html_proc", "running_node_proc"):
            if st.session_state.get(key):
                _kill_process(st.session_state[key])
                st.session_state[key] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Result renderers
# ---------------------------------------------------------------------------

def _render_status_banner(status: str, error: Optional[str], elapsed: float) -> None:
    pill_class = {"success": "status-success", "failed": "status-failed"}.get(
        status, "status-pending"
    )
    label = {"success": "Success", "failed": "Failed"}.get(status, "Incomplete")
    msg = {
        "success": "Agent completed all steps",
        "failed": error or "Unknown error",
    }.get(status, error or "Status unclear")

    st.markdown(
        f"""
        <div class='card'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <span class='status-pill {pill_class}'>{label}</span>
                    <span style='margin-left: 12px;'>{msg}</span>
                </div>
                <span style='color: #6b7280; font-family: monospace; font-size: 0.85rem;'>
                    {elapsed:.1f}s
                </span>
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
        "PLANNER NODE EXECUTING": ("Planner", "#fbbf24"),
        "ORCHESTRATOR AGENT EXECUTING": ("Orchestrator", "#06b6d4"),
        "ROUTER NODE EXECUTING": ("Router", "#06b6d4"),
        "SIMPLE EXECUTOR": ("Simple", "#34d399"),
        "COMPILED EXECUTOR": ("Compiled", "#f472b6"),
        "WEB EXECUTOR": ("Web", "#60a5fa"),
    }
    sequence = []
    for entry in logs:
        for marker, (label, color) in markers.items():
            if marker in entry:
                if not sequence or sequence[-1][0] != label:
                    sequence.append((label, color))
    if not sequence:
        return

    st.markdown("##### Execution Path")
    path_html = "<div class='arch-path'>"
    for i, (label, color) in enumerate(sequence):
        if i > 0:
            path_html += " <span style='color: #6b7280; margin: 0 8px;'>→</span> "
        path_html += f"<span style='color: {color};'>{label}</span>"
    path_html += "</div>"
    st.markdown(path_html, unsafe_allow_html=True)


def _render_plan(plan: list) -> None:
    st.markdown("##### Generated Plan")
    if not plan:
        st.info("No plan generated.")
        return
    plan_html = "<div class='card'>"
    for i, step in enumerate(plan, 1):
        plan_html += (
            f"<div style='display: flex; gap: 12px; margin: 6px 0;'>"
            f"<span style='color: #818cf8; font-weight: 600; min-width: 20px;'>{i}.</span>"
            f"<span>{step}</span></div>"
        )
    plan_html += "</div>"
    st.markdown(plan_html, unsafe_allow_html=True)


def _render_files(files: list, show_contents: bool) -> None:
    st.markdown("##### Generated Files")
    if not files:
        st.info("No files were created.")
        return
    for file_rel_path in files:
        full_path = WORKSPACE / file_rel_path
        with st.expander(f"{file_rel_path}", expanded=False):
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
    st.markdown("##### Recovery Events")
    for issue in issues[:8]:
        st.warning(issue.strip()[:200])
    if len(issues) > 8:
        st.caption(f"... and {len(issues) - 8} more")


def _render_full_trace(logs: list) -> None:
    st.markdown("##### Full Execution Log")
    if not logs:
        st.info("No log entries.")
        return
    log_text = "\n".join(f"{i:>3}. {entry}" for i, entry in enumerate(logs, 1))
    st.code(log_text, language="text")


# ---------------------------------------------------------------------------
# Project launchers
# ---------------------------------------------------------------------------

def _render_web_app_launcher(files: list) -> None:
    st.markdown("---")
    st.markdown("##### FastAPI / Flask Launcher")

    proc = st.session_state.get("running_server_proc")
    port = st.session_state.get("running_server_port", 8000)
    is_running = proc is not None and proc.poll() is None

    if is_running:
        base_url = f"http://localhost:{port}"
        st.markdown(
            f"""
            <div class='card'>
                <span class='status-pill status-success'>● Running</span>
                <span style='margin-left: 12px; font-family: monospace;'>{base_url}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.link_button("Swagger", f"{base_url}/docs", use_container_width=True)
        with col_b:
            st.link_button("Root", f"{base_url}/", use_container_width=True)
        with col_c:
            st.link_button("OpenAPI", f"{base_url}/openapi.json", use_container_width=True)
        with col_d:
            if st.button("Stop", use_container_width=True, key="stop_uvicorn"):
                _kill_process(proc)
                st.session_state["running_server_proc"] = None
                st.session_state["running_server_port"] = None
                time.sleep(0.4)
                st.rerun()

        st.markdown("**Test endpoint**")
        col_path, col_test = st.columns([4, 1])
        with col_path:
            path = st.text_input("path", value="/docs", key="endpoint_path",
                                 label_visibility="collapsed",
                                 placeholder="/weather?city=London")
        with col_test:
            test_clicked = st.button("Test", use_container_width=True, key="test_endpoint")

        if test_clicked and path.strip():
            url = f"{base_url}{path.strip()}"
            try:
                with urllib.request.urlopen(url, timeout=5.0) as resp:
                    body = resp.read(4096).decode("utf-8", errors="replace")
                    st.success(f"HTTP {resp.status} from {url}")
                    try:
                        parsed = json.loads(body)
                        st.code(json.dumps(parsed, indent=2)[:2000], language="json")
                    except (ValueError, json.JSONDecodeError):
                        st.code(body[:2000], language="text")
            except urllib.error.HTTPError as exc:
                st.error(f"HTTP {exc.code} — {exc.reason}")
                try:
                    err_body = exc.read(2000).decode("utf-8", errors="replace")
                    st.code(err_body, language="text")
                except Exception:
                    pass
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                st.error(f"Network error: {exc}")
        return

    app_module = _detect_app_module(files)
    st.markdown(
        f"""
        <div class='card'>
            <p style='color: #6b7280; margin: 0 0 6px 0; font-size: 0.85rem;'>
                Launch the generated web app to test endpoints.
            </p>
            <p style='font-family: monospace; margin: 0; font-size: 0.85rem;'>
                python -m uvicorn {app_module}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Server", type="primary", use_container_width=True, key="launch_uvicorn"):
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
                st.error(f"Server didn't respond within 10s (port {free_port})")
                if stderr:
                    with st.expander("Show stderr"):
                        st.code(stderr[:2000], language="text")
                return
            st.session_state["running_server_proc"] = new_proc
            st.session_state["running_server_port"] = free_port
            time.sleep(0.4)
            st.rerun()


def _render_node_launcher() -> None:
    st.markdown("---")
    st.markdown("##### Node.js Launcher")

    proc = st.session_state.get("running_node_proc")
    port = st.session_state.get("running_node_port", 3000)
    is_running = proc is not None and proc.poll() is None
    run_cmd, entry = _detect_node_entry()

    if is_running:
        base_url = f"http://localhost:{port}"
        st.markdown(
            f"""
            <div class='card'>
                <span class='status-pill status-success'>● Running</span>
                <span style='margin-left: 12px; font-family: monospace;'>{base_url}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_open, col_stop = st.columns([3, 1])
        with col_open:
            st.link_button("Open in new tab", base_url, use_container_width=True)
        with col_stop:
            if st.button("Stop", use_container_width=True, key="stop_node"):
                _kill_process(proc)
                st.session_state["running_node_proc"] = None
                st.session_state["running_node_port"] = None
                time.sleep(0.4)
                st.rerun()

        st.markdown("**Test endpoint**")
        col_path, col_test = st.columns([4, 1])
        with col_path:
            path = st.text_input("node_path", value="/", key="node_endpoint_path",
                                 label_visibility="collapsed", placeholder="/hello")
        with col_test:
            test_clicked = st.button("Test", use_container_width=True, key="test_node_endpoint")

        if test_clicked and path.strip():
            url = f"{base_url}{path.strip()}"
            try:
                with urllib.request.urlopen(url, timeout=5.0) as resp:
                    body = resp.read(4096).decode("utf-8", errors="replace")
                    st.success(f"HTTP {resp.status} from {url}")
                    try:
                        parsed = json.loads(body)
                        st.code(json.dumps(parsed, indent=2)[:2000], language="json")
                    except (ValueError, json.JSONDecodeError):
                        st.code(body[:2000], language="text")
            except urllib.error.HTTPError as exc:
                st.error(f"HTTP {exc.code} — {exc.reason}")
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                st.error(f"Network error: {exc}")
        return

    st.markdown(
        f"""
        <div class='card'>
            <p style='color: #6b7280; margin: 0 0 6px 0; font-size: 0.85rem;'>
                Launch the generated Node.js application.
            </p>
            <p style='font-family: monospace; margin: 0; font-size: 0.85rem;'>
                {run_cmd} → serves <b>{entry}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_install, col_launch = st.columns(2)
    with col_install:
        if st.button("npm install", use_container_width=True, key="npm_install"):
            with st.spinner("Installing dependencies..."):
                rc, stdout, stderr, _ = _run_program("npm install", timeout=120)
                if rc == 0:
                    st.success("Dependencies installed")
                else:
                    st.error("Install failed")
                    st.code(stderr[:1500], language="text")

    with col_launch:
        if st.button("Launch Server", type="primary", use_container_width=True, key="launch_node"):
            with st.spinner("Starting Node server..."):
                # Try common Node ports
                free_port = _find_free_port(3000)
                new_proc = _launch_node_server(run_cmd)
                if new_proc is None:
                    st.error("Failed to launch. Is Node installed?")
                    return
                # Try multiple common ports since we don't know what the code uses
                ready = False
                for candidate in (3000, 8080, 8000, 5000, free_port):
                    if _wait_for_server_ready(candidate, timeout=2.0, paths=("/", "/hello", "/api")):
                        free_port = candidate
                        ready = True
                        break
                if not ready:
                    _kill_process(new_proc)
                    st.error("Node server didn't respond on common ports (3000, 8080, 8000)")
                    return
                st.session_state["running_node_proc"] = new_proc
                st.session_state["running_node_port"] = free_port
                time.sleep(0.4)
                st.rerun()


def _render_html_preview(files: list) -> None:
    st.markdown("---")
    st.markdown("##### HTML Live Preview")

    proc = st.session_state.get("running_html_proc")
    port = st.session_state.get("running_html_port", 8080)
    is_running = proc is not None and proc.poll() is None
    entry_file = _find_html_entry(files) or "index.html"

    if is_running:
        url = f"http://localhost:{port}/{entry_file}"
        st.markdown(
            f"""
            <div class='card'>
                <span class='status-pill status-success'>● Live</span>
                <span style='margin-left: 12px; font-family: monospace;'>{url}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_open, col_stop = st.columns([3, 1])
        with col_open:
            st.link_button("Open in new tab", url, use_container_width=True)
        with col_stop:
            if st.button("Stop", use_container_width=True, key="stop_html"):
                _kill_process(proc)
                st.session_state["running_html_proc"] = None
                st.session_state["running_html_port"] = None
                time.sleep(0.4)
                st.rerun()

        st.markdown("**Preview**")
        st.components.v1.iframe(url, height=600, scrolling=True)
        return

    st.markdown(
        f"""
        <div class='card'>
            <p style='color: #6b7280; margin: 0 0 6px 0; font-size: 0.85rem;'>
                Start a local server to preview the HTML/CSS/JS files.
            </p>
            <p style='font-family: monospace; margin: 0; font-size: 0.85rem;'>
                python -m http.server → serves <b>{entry_file}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Live Server", type="primary",
                 use_container_width=True, key="launch_html"):
        with st.spinner("Starting server..."):
            free_port = _find_free_port(8080)
            new_proc = _launch_html_server(free_port)
            if new_proc is None:
                st.error("Failed to launch HTML server.")
                return
            if not _wait_for_server_ready(free_port, timeout=5.0, paths=("/",)):
                _kill_process(new_proc)
                st.error("HTML server didn't respond.")
                return
            st.session_state["running_html_proc"] = new_proc
            st.session_state["running_html_port"] = free_port
            time.sleep(0.4)
            st.rerun()


def _render_program_simulator(files: list, project_type: str) -> None:
    st.markdown("---")

    if project_type == "python":
        st.markdown("##### Python Simulator")
        py_file = _find_source_file(files, (".py",))
        if not py_file:
            st.info("No Python file found to run.")
            return
        st.markdown(
            f"""
            <div class='card'>
                <p style='color: #6b7280; margin: 0 0 6px 0; font-size: 0.85rem;'>
                    Run the generated Python script with optional stdin input.
                </p>
                <p style='font-family: monospace; margin: 0; font-size: 0.85rem;'>
                    python {py_file}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        command = f"python {py_file}"

    elif project_type == "compiled":
        st.markdown("##### Compiled Program Simulator")
        source_file = _find_source_file(files, (".cpp", ".cc", ".c"))
        if not source_file:
            st.info("No C/C++ source file found.")
            return
        binary_name = Path(source_file).stem
        binary_file = binary_name + (".exe" if IS_WINDOWS else "")
        binary_exists = ((WORKSPACE / binary_file).exists() or
                         (WORKSPACE / binary_name).exists())

        st.markdown(
            f"""
            <div class='card'>
                <p style='color: #6b7280; margin: 0 0 6px 0; font-size: 0.85rem;'>
                    {"Run the compiled binary" if binary_exists else "Compile and run"}.
                </p>
                <p style='font-family: monospace; margin: 0; font-size: 0.85rem;'>
                    {f"./{binary_name}" if binary_exists else f"g++ {source_file} -o {binary_name}"}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not binary_exists:
            if st.button("Compile First", use_container_width=True, key="compile_btn"):
                with st.spinner(f"Compiling {source_file}..."):
                    ok, name, err = _compile_program(source_file)
                    if ok:
                        st.success(f"Compiled → {name}")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Compilation failed")
                        st.code(err, language="text")
            return

        command = f"{binary_name}.exe" if IS_WINDOWS else f"./{binary_name}"

    else:
        return

    st.markdown("**Optional stdin input** (leave empty if not needed)")
    stdin_text = st.text_area(
        "stdin", key=f"stdin_{project_type}", height=70,
        label_visibility="collapsed",
        placeholder="e.g. 5\\n10\\n15  (one value per line)",
    )

    col_run, col_clear = st.columns([4, 1])
    with col_run:
        run_sim = st.button("Run Program", type="primary",
                            use_container_width=True, key=f"run_sim_{project_type}")
    with col_clear:
        if st.button("Clear", use_container_width=True, key=f"clear_sim_{project_type}"):
            st.session_state["simulator_output"] = None
            st.rerun()

    if run_sim:
        with st.spinner(f"Running {command}..."):
            rc, stdout, stderr, elapsed = _run_program(
                command, stdin_input=stdin_text, timeout=15,
            )
            st.session_state["simulator_output"] = {
                "command": command, "exit_code": rc,
                "stdout": stdout, "stderr": stderr, "elapsed": elapsed,
            }

    output = st.session_state.get("simulator_output")
    if output:
        st.markdown("**Output**")
        success = output["exit_code"] == 0
        pill_class = "status-success" if success else "status-failed"
        label = f"Exit {output['exit_code']}"
        st.markdown(
            f"""
            <div class='card' style='padding: 0.6rem 1rem;'>
                <span class='status-pill {pill_class}'>● {label}</span>
                <span style='margin-left: 12px; font-family: monospace; font-size: 0.85rem;'>
                    {output['elapsed']:.2f}s · {output['command']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if output["stdout"]:
            st.markdown("**stdout**")
            st.markdown(
                f"<div class='terminal-box'>{output['stdout'][:5000]}</div>",
                unsafe_allow_html=True,
            )
        if output["stderr"]:
            st.markdown("**stderr**")
            st.markdown(
                f"<div class='terminal-box terminal-error'>{output['stderr'][:5000]}</div>",
                unsafe_allow_html=True,
            )
        if not output["stdout"] and not output["stderr"]:
            st.info("Program produced no output.")


def _render_runners(files: list, logs: list) -> None:
    """Choose which launcher to show based on project type."""
    if _is_node_project(files):
        _render_node_launcher()
    elif _is_web_app_project(files, logs):
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
        f"<div style='color: #6b7280; font-style: italic; margin-bottom: 1rem;'>"
        f"Processing: <span style='color: #818cf8;'>{requirement}</span></div>",
        unsafe_allow_html=True,
    )

    initial_state = AgentState(
        requirement=requirement.strip(),
        plan=[], files=[], logs=[], current_step=0,
        is_complete=False, last_error=None, retry_count=0,
        plan_feedback=None, user_feedback=None,
    )

    with st.spinner("Agent is thinking, generating, and verifying..."):
        try:
            from src.agent.graph import graph
            start_time = time.monotonic()
            result = graph.invoke(initial_state)
            elapsed = time.monotonic() - start_time
            st.session_state["last_result"] = (result, elapsed)
            st.session_state["simulator_output"] = None
        except Exception as exc:
            st.error(f"Fatal error: {exc}")
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
    _render_metrics(plan, files, logs)
    _render_architecture_trace(logs)
    _render_plan(plan)
    _render_files(files, show_files)
    _render_warnings(logs)
    _render_runners(files, logs)

    if show_full_trace:
        _render_full_trace(logs)

elif not run_clicked:
    # Welcome state — three capability cards
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    cards = [
        ("Simple", "#34d399", "Python · HTML · CSS", "Script runner + live preview"),
        ("Compiled", "#f472b6", "C · C++ · Rust · Go", "Auto-compile + stdin"),
        ("Web", "#60a5fa", "FastAPI · Flask · Node", "Launch + test endpoints"),
    ]
    for col, (title, color, langs, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div class='card' style='text-align: center;'>
                    <h3 style='margin: 0 0 0.5rem 0; color: {color};'>{title}</h3>
                    <p style='font-size: 0.85rem; margin: 0.25rem 0; color: #9ca3af;'>{langs}</p>
                    <p style='font-size: 0.75rem; margin: 0.5rem 0 0 0; color: #6b7280;'>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align: center; margin-top: 3rem; color: #6b7280; font-size: 0.8rem;'>
        LangGraph workflow · Orchestrator routing · HTTP verification
    </div>
    """,
    unsafe_allow_html=True,
)