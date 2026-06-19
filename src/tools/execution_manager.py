"""
Execution Manager Module

Responsible for:
- Running shell commands
- Capturing stdout/stderr
- Handling timeouts
- Returning execution results with status codes
"""

import subprocess
from typing import Tuple


def execute_command(command: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """
    Executes a shell command and captures its output.
    
    Args:
        command: Shell command to execute (e.g., "pip install -r requirements.txt")
        timeout: Maximum time to wait for command completion in seconds (default: 30)
    
    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
        - success: True if command returned exit code 0, False otherwise
        - stdout: Standard output from the command
        - stderr: Standard error from the command
    
    Example:
        success, stdout, stderr = execute_command("pip install flask")
        if success:
            print("Command succeeded!")
        else:
            print(f"Command failed: {stderr}")
    """
    print(f"\n--- EXECUTION MANAGER: Running Command ---")
    print(f"Command: {command}")
    print(f"Timeout: {timeout}s")
    
    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,  # Return output as strings, not bytes
            timeout=timeout
        )
        
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
        
        # Print results
        print(f"Return Code: {return_code}")
        print(f"Stdout Length: {len(stdout)} characters")
        print(f"Stderr Length: {len(stderr)} characters")
        
        if stdout:
            print(f"Stdout Preview: {stdout[:200]}...")
        if stderr:
            print(f"Stderr Preview: {stderr[:200]}...")
        
        # Success if return code is 0
        success = return_code == 0
        
        if success:
            print("✓ Command executed successfully")
        else:
            print(f"✗ Command failed with return code {return_code}")
        
        return success, stdout, stderr
    
    except subprocess.TimeoutExpired as e:
        error_msg = f"Command timed out after {timeout} seconds: {command}"
        print(f"✗ {error_msg}")
        return False, "", error_msg
    
    except Exception as e:
        error_msg = f"Failed to execute command: {str(e)}"
        print(f"✗ {error_msg}")
        return False, "", error_msg


def validate_command(command: str) -> Tuple[bool, str]:
    """
    Validates that a command can be run (checks for basic syntax issues).
    
    Args:
        command: Command to validate
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Basic validation
    if not command or not isinstance(command, str):
        return False, "Command must be a non-empty string"
    
    if len(command) > 1000:
        return False, "Command is too long (> 1000 characters)"
    
    return True, ""


# --- Local Testing Block ---
if __name__ == "__main__":
    print("Testing Execution Manager...")
    
    # Test 1: Simple echo command
    print("\n--- Test 1: Echo Command ---")
    success, stdout, stderr = execute_command("echo 'Hello World'")
    print(f"Success: {success}")
    print(f"Output: {stdout}")
    
    # Test 2: Command that fails
    print("\n--- Test 2: Failing Command ---")
    success, stdout, stderr = execute_command("false")
    print(f"Success: {success}")
    print(f"Error: {stderr}")
    
    # Test 3: List files command
    print("\n--- Test 3: List Files ---")
    success, stdout, stderr = execute_command("ls -la")
    print(f"Success: {success}")
    print(f"Output lines: {len(stdout.splitlines())}")
    
    # Test 4: Command with timeout (Windows-compatible)
    print("\n--- Test 4: Timeout Test (should succeed quickly) ---")
    success, stdout, stderr = execute_command("echo 'quick command'", timeout=5)
    print(f"Success: {success}")
