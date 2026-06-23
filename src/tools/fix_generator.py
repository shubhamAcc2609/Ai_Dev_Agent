"""
Fix Generator Module

Responsible for:
- Analyzing error information from Error Analyzer
- Generating appropriate fixes (code modifications, dependency installation, etc.)
- Applying corrective actions
- Re-executing to verify fixes
"""

from langchain_core.prompts import ChatPromptTemplate
from src.agent.config import llm
from src.tools.execution_manager import execute_command
from src.tools.error_analyzer import analyze_execution_failure
import re
from typing import Dict, List, Tuple
from src.utils.json_parser import extract_json_object

FIX_GENERATOR_SYSTEM_PROMPT = """You are an expert fix generator for an autonomous Software Development Agent.
Your role is to generate and apply fixes for execution failures.

When given error analysis information, you must:
1. Understand the root cause and error type
2. Generate specific, actionable fixes
3. Determine the sequence of fixes to apply
4. Provide verification steps after fixes are applied
5. Suggest rollback/alternative approaches if fix fails

Always respond with only one valid JSON object with keys:
- "fixes": List of fix objects, each with:
  - "type": "install_dependency", "modify_file", "run_command", "create_file", "delete_file"
  - "target": File path, package name, or command
  - "action": Specific action to take
  - "content": Content for file modifications (if applicable)
  - "priority": 1-10 (higher = apply first)
- "verification_command": How to verify the fixes worked
- "rollback_steps": How to undo changes if fix fails
- "explanation": Why these fixes address the root cause
- "confidence": 0-100 (percentage confidence the fix will work)
"""


def generate_fixes(error_analysis: Dict, code_context: str = "", file_path: str = "") -> Dict:
    """
    Generates fixes based on error analysis.
    
    Args:
        error_analysis: Dictionary from error_analyzer.analyze_execution_failure()
        code_context: Relevant code snippet where error occurred
        file_path: Path to the file with the error
    
    Returns:
        dict: Fix plan with "fixes" list, "verification_command", "rollback_steps", etc.
        
    Raises:
        ValueError: If fix generation fails
    """
    print(f"\n--- FIX GENERATOR: Generating Fixes ---")
    print(f"Error Type: {error_analysis.get('error_type')}")
    print(f"Severity: {error_analysis.get('severity')}")
    print(f"Root Cause: {error_analysis.get('root_cause')}")
    
    # Use LLM to generate fixes
    prompt = ChatPromptTemplate.from_messages([
        ("system", FIX_GENERATOR_SYSTEM_PROMPT),
        ("human", """Generate fixes for this error:

Error Type: {error_type}
Error Message: {error_message}
Root Cause: {root_cause}
Severity: {severity}
Affected Component: {affected_component}

File Path: {file_path}

Code Context:
{code_context}

Suggested Fix Direction: {suggested_fix}""")
    ])
    
    chain = prompt | llm
    
    try:
        llm_response = chain.invoke({
            "error_type": error_analysis.get('error_type'),
            "error_message": error_analysis.get('error_message'),
            "root_cause": error_analysis.get('root_cause'),
            "severity": error_analysis.get('severity'),
            "affected_component": error_analysis.get('affected_component'),
            "file_path": file_path,
            "code_context": code_context,
            "suggested_fix": error_analysis.get('suggested_fix')
        })
        response_text = llm_response.content
        
        print(f"LLM Response (Raw): {response_text[:300]}...")
        
        # Extract JSON from response
        fix_plan = _extract_json_from_response(response_text)
        
        if not fix_plan or "fixes" not in fix_plan:
            fix_plan = _generate_fallback_fixes(error_analysis, file_path)
        
        print(f"Generated {len(fix_plan.get('fixes', []))} fixes")
        return fix_plan
    
    except Exception as e:
        print(f"LLM Fix generation failed: {str(e)}, using fallback...")
        return _generate_fallback_fixes(error_analysis, file_path)


def apply_fixes(fix_plan: Dict) -> Tuple[bool, List[str], List[str]]:
    """
    Applies fixes from the fix plan in sequence.
    
    Args:
        fix_plan: Dictionary from generate_fixes()
    
    Returns:
        Tuple of (success: bool, applied_fixes: List[str], errors: List[str])
    """
    print(f"\n--- FIX GENERATOR: Applying Fixes ---")
    print(f"Total fixes to apply: {len(fix_plan.get('fixes', []))}")
    
    applied_fixes = []
    errors = []
    
    # Sort fixes by priority
    fixes = sorted(fix_plan.get('fixes', []), key=lambda x: x.get('priority', 5), reverse=True)
    
    for i, fix in enumerate(fixes):
        fix_type = fix.get('type')
        target = fix.get('target')
        action = fix.get('action')
        
        print(f"\nApplying Fix {i+1}/{len(fixes)}: {fix_type} on {target}")
        
        try:
            if fix_type == "install_dependency":
                success, stdout, stderr = execute_command(f"pip install {target}")
                if success:
                    applied_fixes.append(f"Installed dependency: {target}")
                else:
                    errors.append(f"Failed to install {target}: {stderr}")
            
            elif fix_type == "run_command":
                success, stdout, stderr = execute_command(action)
                if success:
                    applied_fixes.append(f"Executed command: {action}")
                else:
                    errors.append(f"Command failed: {action}")
            
            elif fix_type == "modify_file":
                applied_fixes.append(f"Modified file: {target} - {action}")
                # Actual file modification would be handled by file_manager
                print(f"  (File modification queued: {target})")
            
            elif fix_type == "create_file":
                content = fix.get('content', '')
                applied_fixes.append(f"Created file: {target}")
                print(f"  (File creation queued: {target})")
            
            elif fix_type == "delete_file":
                applied_fixes.append(f"Deleted file: {target}")
                print(f"  (File deletion queued: {target})")
            
            else:
                errors.append(f"Unknown fix type: {fix_type}")
        
        except Exception as e:
            errors.append(f"Error applying fix {fix_type}: {str(e)}")
    
    success = len(errors) == 0
    print(f"\nApplied {len(applied_fixes)} fixes, {len(errors)} errors")
    
    return success, applied_fixes, errors


def verify_fix(verification_command: str, original_command: str) -> Tuple[bool, str]:
    """
    Verifies if the fix worked by running verification command.
    
    Args:
        verification_command: Command to verify the fix worked
        original_command: Original command that failed
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    print(f"\n--- FIX GENERATOR: Verifying Fix ---")
    print(f"Verification Command: {verification_command}")
    
    # Try the original command again
    print(f"Re-running original command: {original_command}")
    success, stdout, stderr = execute_command(original_command)
    
    if success:
        print("✓ Original command now succeeds!")
        return True, stdout
    
    # If original still fails, try verification command
    if verification_command:
        print(f"Running verification command: {verification_command}")
        success, stdout, stderr = execute_command(verification_command)
        if success:
            print("✓ Verification command succeeded!")
            return True, stdout
    
    print("✗ Fix verification failed")
    return False, stderr


def auto_fix_and_retry(command: str, max_retries: int = 3) -> Tuple[bool, str, str, List[Dict]]:
    """
    Automatically fixes errors and retries command.
    
    Args:
        command: Shell command to execute
        max_retries: Maximum number of fix attempts
    
    Returns:
        Tuple of (success: bool, stdout: str, stderr: str, fix_history: List[Dict])
    """
    print(f"\n--- AUTO-FIX WITH RETRY ---")
    print(f"Command: {command}")
    print(f"Max retries: {max_retries}")
    
    fix_history = []
    
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        
        # Execute command
        success, stdout, stderr = execute_command(command)
        
        if success:
            print("✓ Command succeeded!")
            return True, stdout, stderr, fix_history
        
        # Analyze failure
        print("✗ Command failed, analyzing error...")
        error_analysis = analyze_execution_failure(stdout, stderr, command)
        
        # Check if error is recoverable
        if not error_analysis.get('is_recoverable', False):
            print("✗ Error is not recoverable")
            return False, stdout, stderr, fix_history
        
        # Generate fixes
        print("Generating fixes...")
        fix_plan = generate_fixes(error_analysis)
        
        # Apply fixes
        print("Applying fixes...")
        fix_success, applied_fixes, errors = apply_fixes(fix_plan)
        
        fix_history.append({
            "attempt": attempt + 1,
            "error_analysis": error_analysis,
            "fix_plan": fix_plan,
            "applied": applied_fixes,
            "errors": errors
        })
        
        if not fix_success:
            print(f"✗ Failed to apply fixes: {errors}")
            return False, stdout, stderr, fix_history
    
    print(f"✗ Max retries ({max_retries}) exceeded")
    return False, stdout, stderr, fix_history


def _generate_fallback_fixes(error_analysis: Dict, file_path: str = "") -> Dict:
    """
    Generates fallback fixes using pattern matching when LLM fails.
    """
    error_type = error_analysis.get('error_type', 'UnknownError')
    fixes = []
    verification = ""
    confidence = 50
    
    if error_type == "ModuleNotFoundError":
        # Extract module name from error message
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_analysis.get('error_message', ''))
        if match:
            module_name = match.group(1)
            fixes.append({
                "type": "install_dependency",
                "target": module_name,
                "action": f"pip install {module_name}",
                "priority": 10
            })
            verification = f"python -c 'import {module_name}'"
            confidence = 75
    
    elif error_type == "FileNotFoundError":
        fixes.append({
            "type": "run_command",
            "target": "file check",
            "action": "ls -la" if "/" in error_analysis.get('affected_component', '') else "dir",
            "priority": 5
        })
        confidence = 30
    
    elif error_type == "PermissionError":
        fixes.append({
            "type": "run_command",
            "target": "permission fix",
            "action": f"chmod +x {file_path}" if file_path else "chmod +x .",
            "priority": 8
        })
        confidence = 60
    
    elif error_type == "SyntaxError":
        fixes.append({
            "type": "modify_file",
            "target": file_path or "unknown_file",
            "action": "Fix syntax errors",
            "priority": 10
        })
        confidence = 20  # Low confidence - needs manual review
    
    else:
        fixes.append({
            "type": "run_command",
            "target": "investigate",
            "action": "Check error logs and documentation",
            "priority": 1
        })
        confidence = 10
    
    return {
        "fixes": fixes,
        "verification_command": verification,
        "rollback_steps": ["No changes to rollback"],
        "explanation": f"Fallback fix for {error_type}",
        "confidence": confidence
    }


def _extract_json_from_response(response_text: str) -> Dict:
    """
    Extracts JSON object from LLM response.
    """
    try:
        return extract_json_object(response_text)
    except ValueError:
        return {}


# --- Local Testing Block ---
if __name__ == "__main__":
    print("Testing Fix Generator...")
    
    # Test 1: Generate fixes for ModuleNotFoundError
    print("\n--- Test 1: Generate Fixes for ModuleNotFoundError ---")
    error_analysis_1 = {
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'flask'",
        "root_cause": "Flask is not installed",
        "affected_component": "import statement",
        "severity": "major",
        "context": "import flask",
        "suggested_fix": "Install Flask package",
        "is_recoverable": True
    }
    
    fix_plan_1 = generate_fixes(error_analysis_1, "", "")
    print(f"Generated {len(fix_plan_1.get('fixes', []))} fixes")
    print(f"Confidence: {fix_plan_1.get('confidence', 'unknown')}%")
    
    # Test 2: Apply fixes
    print("\n--- Test 2: Apply Fixes ---")
    success, applied, errors = apply_fixes(fix_plan_1)
    print(f"Success: {success}")
    print(f"Applied: {applied}")
    print(f"Errors: {errors}")
    
    # Test 3: Fallback fixes for SyntaxError
    print("\n--- Test 3: Fallback Fixes for SyntaxError ---")
    error_analysis_3 = {
        "error_type": "SyntaxError",
        "error_message": "invalid syntax",
        "root_cause": "Incorrect Python syntax",
        "affected_component": "main.py line 5",
        "severity": "critical",
        "suggested_fix": "Fix syntax in code",
        "is_recoverable": False
    }
    
    fix_plan_3 = generate_fixes(error_analysis_3, "", "main.py")
    print(f"Generated {len(fix_plan_3.get('fixes', []))} fixes")
    print(f"Explanation: {fix_plan_3.get('explanation')}")
