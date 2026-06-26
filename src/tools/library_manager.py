"""
Library Manager

Single source of truth for "what's installed on this host?"

Used by:
- Planner: to avoid generating plans that depend on missing toolchains
- UI: to show host-environment status to the user
- Executors: as a defense-in-depth check before invoking external tools

Design principles:
- Cheap: results cached in-memory after first detection
- Lazy: only checks tools when asked
- Honest: returns False if uncertain, never lies about availability
- Extensible: adding a new tool is one line in TOOLCHAINS
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Toolchain catalog
# ---------------------------------------------------------------------------

@dataclass
class Toolchain:
    """Definition of a toolchain we know how to detect."""
    name: str                          # Executable name to check
    label: str                         # Human-readable name for logs/UI
    category: str                      # "python", "node", "compiled", etc.
    install_hint: str = ""             # How the user can install it
    aliases: List[str] = field(default_factory=list)  # Other names to try


# Canonical list of toolchains we care about.
# Adding a new one: just append here. Library Manager will pick it up.
TOOLCHAINS: List[Toolchain] = [
    # ─── Python ecosystem ─────────────────────────────────────────────
    Toolchain("python", "Python", "python",
              install_hint="https://python.org/downloads",
              aliases=["python3", "py"]),
    Toolchain("pip", "pip", "python",
              install_hint="Bundled with Python 3.4+",
              aliases=["pip3"]),

    # ─── Node ecosystem ───────────────────────────────────────────────
    Toolchain("node", "Node.js", "node",
              install_hint="winget install OpenJS.NodeJS",
              aliases=[]),
    Toolchain("npm", "npm", "node",
              install_hint="Bundled with Node.js",
              aliases=[]),
    Toolchain("yarn", "Yarn", "node",
              install_hint="npm install -g yarn",
              aliases=[]),
    Toolchain("pnpm", "pnpm", "node",
              install_hint="npm install -g pnpm",
              aliases=[]),

    # ─── C / C++ ──────────────────────────────────────────────────────
    Toolchain("gcc", "GCC", "compiled",
              install_hint="MSYS2: pacman -S mingw-w64-ucrt-x86_64-gcc",
              aliases=[]),
    Toolchain("g++", "G++", "compiled",
              install_hint="MSYS2: pacman -S mingw-w64-ucrt-x86_64-gcc",
              aliases=[]),
    Toolchain("clang", "Clang", "compiled",
              install_hint="winget install LLVM.LLVM",
              aliases=[]),

    # ─── Other compiled languages ─────────────────────────────────────
    Toolchain("rustc", "Rust", "compiled",
              install_hint="https://rustup.rs",
              aliases=[]),
    Toolchain("cargo", "Cargo", "compiled",
              install_hint="Bundled with rustup",
              aliases=[]),
    Toolchain("go", "Go", "compiled",
              install_hint="https://go.dev/dl",
              aliases=[]),
    Toolchain("javac", "Java Compiler", "compiled",
              install_hint="winget install Microsoft.OpenJDK.21",
              aliases=[]),
    Toolchain("java", "Java Runtime", "compiled",
              install_hint="winget install Microsoft.OpenJDK.21",
              aliases=[]),
    Toolchain("dotnet", ".NET", "compiled",
              install_hint="winget install Microsoft.DotNet.SDK.8",
              aliases=[]),

    # ─── Version control ──────────────────────────────────────────────
    Toolchain("git", "Git", "vcs",
              install_hint="winget install Git.Git",
              aliases=[]),

    # ─── Containers ───────────────────────────────────────────────────
    Toolchain("docker", "Docker", "container",
              install_hint="winget install Docker.DockerDesktop",
              aliases=[]),
]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# Module-level cache so we don't re-check every Planner invocation.
# Format: {tool_name: (is_available, resolved_path, version_string)}
_DETECTION_CACHE: Dict[str, tuple] = {}


def _detect_one(toolchain: Toolchain) -> tuple:
    """
    Check if a single toolchain is available.

    Returns: (is_available, resolved_path, version_string)
    """
    # Try the primary name first, then aliases
    candidates = [toolchain.name] + toolchain.aliases

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            version = _get_version(candidate)
            logger.debug("Detected %s at %s (version=%s)",
                         toolchain.label, path, version)
            return True, path, version

    return False, "", ""


def _get_version(executable: str) -> str:
    """
    Try to extract a version string by running `tool --version`.
    Returns the first line or empty string on failure.
    """
    for flag in ("--version", "-version", "-V"):
        try:
            result = subprocess.run(
                [executable, flag],
                capture_output=True,
                text=True,
                timeout=3,
                encoding="utf-8",
                errors="replace",
            )
            output = (result.stdout or result.stderr or "").strip()
            if output:
                first_line = output.split("\n")[0].strip()
                return first_line[:80]
        except (subprocess.TimeoutExpired, OSError):
            continue
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available(tool_name: str) -> bool:
    """
    Check if a specific tool is installed.

    Cached after first check. Returns False on any uncertainty.
    """
    if tool_name in _DETECTION_CACHE:
        return _DETECTION_CACHE[tool_name][0]

    # Find the matching Toolchain definition
    tc = next((t for t in TOOLCHAINS if t.name == tool_name), None)
    if tc is None:
        # Unknown tool — do a simple shutil.which check
        path = shutil.which(tool_name)
        result = (bool(path), path or "", "")
        _DETECTION_CACHE[tool_name] = result
        return result[0]

    result = _detect_one(tc)
    _DETECTION_CACHE[tool_name] = result
    return result[0]


def get_status(tool_name: str) -> Dict[str, str]:
    """Get full status for a tool: availability, path, version, hint."""
    if tool_name not in _DETECTION_CACHE:
        is_available(tool_name)  # Trigger detection

    available, path, version = _DETECTION_CACHE.get(tool_name, (False, "", ""))
    tc = next((t for t in TOOLCHAINS if t.name == tool_name), None)

    return {
        "name": tool_name,
        "label": tc.label if tc else tool_name,
        "category": tc.category if tc else "unknown",
        "available": available,
        "path": path,
        "version": version,
        "install_hint": tc.install_hint if tc else "",
    }


def detect_all() -> Dict[str, Dict]:
    """
    Detect all known toolchains. Returns dict keyed by tool name.
    Cached results returned on subsequent calls.
    """
    results = {}
    for tc in TOOLCHAINS:
        results[tc.name] = get_status(tc.name)
    return results


def get_available_tools() -> List[str]:
    """Return list of tool names that are currently available."""
    return [tc.name for tc in TOOLCHAINS if is_available(tc.name)]


def get_missing_tools() -> List[str]:
    """Return list of tool names that are NOT available."""
    return [tc.name for tc in TOOLCHAINS if not is_available(tc.name)]


def reset_cache() -> None:
    """Clear the detection cache. Useful for tests or after installing tools."""
    _DETECTION_CACHE.clear()


# ---------------------------------------------------------------------------
# Planner-facing summary
# ---------------------------------------------------------------------------

def get_planner_context() -> str:
    """
    Build a string the Planner can drop into its prompt.

    Format is concise and prescriptive — tells the Planner exactly which
    languages/frameworks it can and cannot use.
    """
    available = get_available_tools()
    missing = get_missing_tools()

    # Group by category for readability
    available_by_cat: Dict[str, List[str]] = {}
    missing_by_cat: Dict[str, List[str]] = {}

    for tc in TOOLCHAINS:
        bucket = available_by_cat if tc.name in available else missing_by_cat
        bucket.setdefault(tc.category, []).append(tc.label)

    lines = ["HOST TOOLCHAIN STATUS:"]
    lines.append("")
    lines.append("✓ AVAILABLE (you may use these):")
    if available_by_cat:
        for cat, tools in sorted(available_by_cat.items()):
            lines.append(f"  {cat}: {', '.join(tools)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("✗ NOT INSTALLED (do NOT plan with these):")
    if missing_by_cat:
        for cat, tools in sorted(missing_by_cat.items()):
            lines.append(f"  {cat}: {', '.join(tools)}")
    else:
        lines.append("  (none — all checked tools are available)")

    lines.append("")
    lines.append(
        "CRITICAL: If a tool is marked NOT INSTALLED, your plan MUST NOT "
        "include any step that depends on it. Instead, choose a different "
        "language/framework from the AVAILABLE list. If the user's request "
        "specifically demands an unavailable tool, generate a plan that "
        "explains the limitation rather than failing silently."
    )

    return "\n".join(lines)


