"""
Error Analyzer Module

Responsible for:
- Examining execution failures from stdout/stderr
- Identifying root causes and error patterns
- Classifying error types (syntax, runtime, dependency, permission, timeout, etc.)
- Extracting relevant error context for fix generation
"""

from langchain_core.prompts import ChatPromptTemplate
from src.agent.config import llm
import re
from typing import Dict, List, Tuple
from src.utils.json_parser import extract_json_object

ERROR_ANALYZER_SYSTEM_PROMPT = """You are an expert error analysis system for an autonomous Software Development Agent.
Your role is to examine execution failures and identify root causes.

When given execution output (stdout/stderr), you must:
1. Identify the error type (syntax, runtime, dependency, permission, timeout, import, etc.)
2. Extract the error message and relevant context
3. Identify which line/component failed
4. Suggest the root cause
5. Estimate severity (critical, major, minor)
6. Recommend next steps for fixing

Always respond with only one valid JSON object with keys:
- "error_type": Category of error (e.g., "ModuleNotFoundError", "SyntaxError", "PermissionError", "TimeoutError")
- "error_message": The actual error message from the output
- "root_cause": Analysis of why this error occurred
- "affected_component": What part of the code/system failed
- "severity": "critical", "major", or "minor"
- "context": Relevant lines or context around the error
- "suggested_fix": What needs to be done to resolve it
- "is_recoverable": True if the error can be automatically fixed, False otherwise
"""


def analyze_execution_failure(stdout: str, stderr: str, command: str = "") -> Dict:
    """
    Analyzes execution failure by examining stdout/stderr.
    
    Args:
        stdout: Standard output from failed command
        stderr: Standard error from failed command
        command: The command that was executed (for context)
    
    Returns:
        dict: Analysis result with keys like "error_type", "root_cause", "severity", etc.
        
    Raises:
        ValueError: If analysis fails
    """
    print(f"\n--- ERROR ANALYZER: Analyzing Failure ---")
    print(f"Command: {command}")
    print(f"Stdout Length: {len(stdout)}")
    print(f"Stderr Length: {len(stderr)}")
    
    # Combine output for analysis
    combined_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    
    if not stderr and not stdout:
        return {
            "error_type": "UnknownError",
            "error_message": "No output captured",
            "root_cause": "Command failed but no error details were captured",
            "affected_component": "Unknown",
            "severity": "major",
            "context": "",
            "suggested_fix": "Run the command with verbose output or debugging enabled",
            "is_recoverable": False
        }
    
    # Use LLM to analyze the error
    prompt = ChatPromptTemplate.from_messages([
        ("system", ERROR_ANALYZER_SYSTEM_PROMPT),
        ("human", """Analyze this execution failure:

Command: {command}

Output:
{combined_output}""")
    ])
    
    chain = prompt | llm
    
    try:
        llm_response = chain.invoke({
            "command": command,
            "combined_output": combined_output
        })
        response_text = llm_response.content
        
        print(f"LLM Analysis (Raw): {response_text[:300]}...")
        
        # Extract JSON from response
        analysis = _extract_json_from_response(response_text)
        
        # Fallback analysis if LLM fails
        if not analysis or "error_type" not in analysis:
            analysis = _fallback_error_analysis(stdout, stderr, command)
        
        print(f"Error Analysis Complete: Type={analysis.get('error_type')}, Severity={analysis.get('severity')}")
        return analysis
    
    except Exception as e:
        print(f"LLM Analysis failed: {str(e)}, using fallback...")
        return _fallback_error_analysis(stdout, stderr, command)


def classify_error_type(output: str) -> str:
    """
    Classifies error type from output using pattern matching.
    
    Args:
        output: Combined stdout/stderr
    
    Returns:
        str: Error type classification
    """
    # Error patterns mapping
    patterns = {
        "ModuleNotFoundError": r"(ModuleNotFoundError|ImportError|No module named)",
        "SyntaxError": r"SyntaxError|IndentationError",
        "TypeError": r"TypeError",
        "AttributeError": r"AttributeError",
        "ValueError": r"ValueError",
        "KeyError": r"KeyError",
        "FileNotFoundError": r"FileNotFoundError|No such file",
        "PermissionError": r"PermissionError|permission denied|Access denied",
        "TimeoutError": r"TimeoutError|timeout",
        "ConnectionError": r"ConnectionError|Connection refused|Failed to connect",
        "DependencyError": r"Could not find|requirement not satisfied",
        "RuntimeError": r"RuntimeError",
        "OSError": r"OSError",
    }
    
    for error_type, pattern in patterns.items():
        if re.search(pattern, output, re.IGNORECASE):
            return error_type
    
    return "UnclassifiedError"


def extract_error_context(output: str, window_size: int = 3) -> str:
    """
    Extracts relevant error context from output.
    
    Args:
        output: Combined stdout/stderr
        window_size: Number of lines before/after error to include
    
    Returns:
        str: Relevant error context
    """
    lines = output.split("\n")
    
    # Find error indicators
    error_indicators = ["error", "exception", "traceback", "failed"]
    error_line_index = -1
    
    for i, line in enumerate(lines):
        if any(indicator in line.lower() for indicator in error_indicators):
            error_line_index = i
            break
    
    if error_line_index == -1:
        # Return last few lines if no specific error found
        return "\n".join(lines[-window_size:])
    
    # Extract context window
    start = max(0, error_line_index - window_size)
    end = min(len(lines), error_line_index + window_size + 1)
    
    return "\n".join(lines[start:end])


def _fallback_error_analysis(stdout: str, stderr: str, command: str) -> Dict:
    """
    Fallback error analysis using pattern matching when LLM fails.
    """
    combined_output = f"{stdout}\n{stderr}"
    error_type = classify_error_type(combined_output)
    error_context = extract_error_context(combined_output)
    
    return {
        "error_type": error_type,
        "error_message": stderr.split("\n")[0] if stderr else stdout.split("\n")[0],
        "root_cause": f"Automatic analysis detected {error_type}",
        "affected_component": "Unknown (requires manual inspection)",
        "severity": "major",
        "context": error_context,
        "suggested_fix": f"Investigate {error_type} - check error message above",
        "is_recoverable": error_type in ["ModuleNotFoundError", "DependencyError", "FileNotFoundError"]
    }


def _extract_json_from_response(response_text: str) -> Dict:
    """
    Extracts JSON object from LLM response.
    
    LLM might respond with explanation text around the JSON.
    """
    try:
        return extract_json_object(response_text)
    except ValueError:
        return {}


def get_error_severity_score(analysis: Dict) -> int:
    """
    Converts severity to numeric score for sorting.
    
    Args:
        analysis: Error analysis dictionary
    
    Returns:
        int: Score (3=critical, 2=major, 1=minor)
    """
    severity_map = {
        "critical": 3,
        "major": 2,
        "minor": 1
    }
    return severity_map.get(analysis.get("severity", "major"), 2)


# --- Local Testing Block ---
if __name__ == "__main__":
    print("Testing Error Analyzer...")
    
    # Test 1: Module not found error
    print("\n--- Test 1: ModuleNotFoundError ---")
    stdout_1 = ""
    stderr_1 = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'"""
    
    analysis_1 = analyze_execution_failure(stdout_1, stderr_1, "python test.py")
    print(f"Detected Error Type: {analysis_1.get('error_type')}")
    print(f"Severity: {analysis_1.get('severity')}")
    print(f"Recoverable: {analysis_1.get('is_recoverable')}")
    
    # Test 2: Syntax error
    print("\n--- Test 2: SyntaxError ---")
    stderr_2 = """File "main.py", line 5
    if x = 5
       ^
SyntaxError: invalid syntax"""
    
    analysis_2 = analyze_execution_failure("", stderr_2, "python main.py")
    print(f"Detected Error Type: {analysis_2.get('error_type')}")
    print(f"Root Cause: {analysis_2.get('root_cause')}")
    
    # Test 3: Permission denied
    print("\n--- Test 3: PermissionError ---")
    stderr_3 = "Permission denied: /root/protected_file.txt"
    
    analysis_3 = analyze_execution_failure("", stderr_3, "cat /root/protected_file.txt")
    print(f"Detected Error Type: {analysis_3.get('error_type')}")
    print(f"Severity: {analysis_3.get('severity')}")
    
    # Test 4: Dependency error
    print("\n--- Test 4: DependencyError ---")
    stderr_4 = """ERROR: Could not find a version that satisfies the requirement numpy==999.0.0
ERROR: No matching distribution found for numpy==999.0.0"""
    
    analysis_4 = analyze_execution_failure("", stderr_4, "pip install numpy==999.0.0")
    print(f"Detected Error Type: {analysis_4.get('error_type')}")
    print(f"Suggested Fix: {analysis_4.get('suggested_fix')}")
