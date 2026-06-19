"""
Executor Node

Orchestrates the execution of a single step in the plan by:
1. Using Code Generator to create an execution plan
2. Using File Manager to create/update files
3. Using Execution Manager to run commands
4. Updating the agent state based on results
"""

from src.agent.state import AgentState
from src.tools.code_generator import generate_execution_plan
from src.tools.file_manager import create_or_update_file
from src.tools.execution_manager import execute_command


def executor_node(state: AgentState) -> dict:
    """
    Executes the current step in the plan and updates state accordingly.
    
    Orchestration Flow:
    1. Check if all steps are complete
    2. Get the current step description
    3. Code Generator: Ask LLM to generate execution plan
    4. File Manager: Create/update files if needed
    5. Execution Manager: Run commands if needed
    6. Update state: logs, files, retry_count, current_step, is_complete
    
    Args:
        state: AgentState containing plan, current_step, logs, files, etc.
    
    Returns:
        dict: State updates to be merged back into the main state
    """
    print("---  EXECUTOR NODE EXECUTING ---")
    
    # ========== STEP 1: Extract Current State ==========
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    logs = state.get("logs", [])
    files = state.get("files", [])
    retry_count = state.get("retry_count", 0)
    
    # ========== STEP 2: Check if All Steps Complete ==========
    if current_step >= len(plan):
        log_message = f"Executor Node: All {len(plan)} steps completed successfully!"
        print(log_message)
        return {
            "logs": [log_message],
            "is_complete": True,
            "current_step": current_step,
            "retry_count": 0,
            "plan_feedback": None,
            "last_error": None,
        }
    
    # ========== STEP 3: Get Current Step Description ==========
    step_description = plan[current_step]
    log_message = f"Executor Node: Executing step {current_step + 1}/{len(plan)}: {step_description}"
    print(log_message)
    logs.append(log_message)
    
    try:
        # ========== STEP 4: Code Generator - Generate Execution Plan ==========
        print("\n[1/3] Invoking Code Generator...")
        execution_plan = generate_execution_plan(step_description)
        print(f"✓ Code Generator returned plan with keys: {list(execution_plan.keys())}")
        logs.append(f"Code Generator: Generated execution plan with keys {list(execution_plan.keys())}")
        
        # ========== STEP 5: File Manager - Create/Update Files ==========
        files_created_this_step = []
        
        if "file_path" in execution_plan and "file_content" in execution_plan:
            print("\n[2/3] Invoking File Manager...")
            file_path = execution_plan["file_path"]
            file_content = execution_plan["file_content"]
            
            success, error, files_created = create_or_update_file(file_path, file_content)
            
            if success:
                print(f"✓ File Manager created: {files_created}")
                files_created_this_step.extend(files_created)
                files.extend(files_created)
                logs.append(f"File Manager: Successfully created {files_created}")
            else:
                print(f"✗ File Manager failed: {error}")
                logs.append(f"File Manager: Failed - {error}")
                # File creation failed - this is a step failure
                return _handle_step_failure(
                    logs, retry_count, current_step, step_description, 
                    f"File creation failed: {error}"
                )
        else:
            print("\n[2/3] File Manager: No file creation needed")
            logs.append("File Manager: Skipped (no file_path/file_content in plan)")
        
        # ========== STEP 6: Execution Manager - Run Commands ==========
        if "command" in execution_plan:
            print("\n[3/3] Invoking Execution Manager...")
            command = execution_plan["command"]
            
            success, stdout, stderr = execute_command(command, timeout=30)
            
            if success:
                print(f"✓ Execution Manager succeeded")
                output_summary = f"Stdout: {stdout[:200] if stdout else '(empty)'}"
                logs.append(f"Execution Manager: Command succeeded. {output_summary}")
            else:
                print(f"✗ Execution Manager failed: {stderr}")
                logs.append(f"Execution Manager: Command failed. Error: {stderr}")
                # Command failed - this is a step failure
                return _handle_step_failure(
                    logs, retry_count, current_step, step_description,
                    f"Command failed: {stderr}"
                )
        else:
            print("\n[3/3] Execution Manager: No command needed")
            logs.append("Execution Manager: Skipped (no command in plan)")
        
        # ========== STEP 7: Success - Move to Next Step ==========
        print("\n✓ Step completed successfully!")
        log_message = (
            f"Executor Node: Step {current_step + 1} completed successfully. "
            f"Files created: {files_created_this_step if files_created_this_step else 'None'}"
        )
        logs.append(log_message)
        
        return {
            "logs": logs,
            "files": files,
            "current_step": current_step + 1,
            "retry_count": 0,
            "last_error": None,
            "plan_feedback": None,
            "is_complete": False,
        }
    
    except Exception as e:
        # ========== Catastrophic Failure (LLM error, parsing error, etc.) ==========
        error_msg = str(e)
        log_message = f"Executor Node: CRITICAL ERROR - {error_msg}"
        print(f"✗ {log_message}")
        logs.append(log_message)
        
        return _handle_step_failure(
            logs, retry_count, current_step, step_description, error_msg
        )


def _handle_step_failure(logs: list, retry_count: int, current_step: int, 
                         step_description: str, error_msg: str) -> dict:
    """
    Helper function to handle step failures with retry logic.
    
    If retries exhausted: Send feedback to planner for replanning
    If retries available: Retry the same step
    
    Args:
        logs: Current logs list
        retry_count: Current retry count
        current_step: Current step index
        step_description: The step that failed
        error_msg: Error message
    
    Returns:
        dict: State updates for retry or replanning
    """
    retry_count += 1
    max_retries = 3
    
    if retry_count >= max_retries:
        # ========== Max Retries Exceeded - Ask Planner to Replan ==========
        log_message = (
            f"Executor Node: Step {current_step + 1} failed after {max_retries} retries. "
            f"Error: {error_msg}. Requesting replanning..."
        )
        print(f"✗ {log_message}")
        logs.append(log_message)
        
        return {
            "logs": logs,
            "retry_count": retry_count,
            "last_error": error_msg,
            "plan_feedback": f"Step '{step_description}' failed: {error_msg}. Please provide an alternative approach.",
            "is_complete": False,
        }
    else:
        # ========== Retries Available - Retry Same Step ==========
        log_message = (
            f"Executor Node: Step {current_step + 1} failed (retry {retry_count}/{max_retries}). "
            f"Error: {error_msg}. Retrying..."
        )
        print(f"⚠ {log_message}")
        logs.append(log_message)
        
        return {
            "logs": logs,
            "retry_count": retry_count,
            "last_error": error_msg,
            "plan_feedback": None,
            "is_complete": False,
        }


# --- Local Testing Block ---
if __name__ == "__main__":
    mock_state = AgentState(
        requirement="Create a simple Python FastAPI for weather endpoints.",
        plan=[
            "Create a requirements.txt with FastAPI and uvicorn",
            "Create a basic main.py with a health check endpoint",
            "Test the API with curl or requests"
        ],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False,
        last_error=None,
        retry_count=0,
        plan_feedback=None,
        user_feedback=None
    )
    
    print("Testing Executor Node (Step 1)...")
    result = executor_node(mock_state)
    print("\n--- Executor Result ---")
    for key, value in result.items():
        if key == "logs":
            print(f"{key}:")
            for log in value:
                print(f"  - {log}")
        else:
            print(f"{key}: {value}")
