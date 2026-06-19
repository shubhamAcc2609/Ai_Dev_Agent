"""
Code Generator Module

Responsible for:
- Taking a step description
- Using LLM to generate structured execution instructions
- Returning a JSON execution plan with file paths, content, and commands
"""

from langchain_core.prompts import ChatPromptTemplate
from src.agent.config import llm
import json

CODE_GENERATOR_SYSTEM_PROMPT = """You are an expert code executor for an autonomous Software Development Agent.
Your role is to execute a specific implementation step and return structured feedback.

When given a step description, you must:
1. Understand what the step requires (e.g., "Create requirements.txt with Flask")
2. Generate the exact commands/code to execute that step
3. Suggest how to verify the step was successful
4. Provide error handling guidance if something goes wrong

Always respond in JSON format with keys:
- "command": The shell command or code to execute (optional)
- "file_path": If creating/modifying a file, the relative path (optional)
- "file_content": If creating/modifying a file, the exact content (optional)
- "verification": How to verify this step worked (optional)
- "description": Brief explanation of what this step does
"""


def generate_execution_plan(step_description: str) -> dict:
    """
    Takes a step description and uses LLM to generate structured execution instructions.
    
    Args:
        step_description: Human-readable task description (e.g., "Create requirements.txt with Flask")
    
    Returns:
        dict: Execution plan with keys like "command", "file_path", "file_content", "verification"
        
    Raises:
        ValueError: If LLM response is not valid JSON or API fails
    """
    print(f"\n--- CODE GENERATOR: Processing Step ---")
    print(f"Step: {step_description}")
    
    # Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", CODE_GENERATOR_SYSTEM_PROMPT),
        ("human", "Execute this step: {step_description}")
    ])
    
    # Create chain: prompt → LLM
    chain = prompt | llm
    
    try:
        # Invoke LLM to get execution plan
        llm_response = chain.invoke({"step_description": step_description})
        response_text = llm_response.content
        
        print(f"LLM Response (Raw): {response_text[:200]}...")
        
        # Extract JSON from response (LLM might wrap it in explanation text)
        execution_plan = _extract_json_from_response(response_text)
        
        print(f"Parsed Execution Plan: {execution_plan}")
        return execution_plan
    
    except Exception as e:
        raise ValueError(f"Code Generator failed: {str(e)}")


def _extract_json_from_response(response_text: str) -> dict:
    """
    Extracts JSON object from LLM response.
    
    LLM might respond with explanation text around the JSON, like:
    "Here's how to do it: { ... } Let me know if you need clarification."
    
    This function extracts just the JSON part.
    
    Args:
        response_text: Raw LLM response text
    
    Returns:
        dict: Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON found in response
    """
    try:
        # Look for JSON object in the response
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            
            execution_plan = json.loads(json_str)
            return execution_plan
        else:
            raise ValueError("No JSON object found in LLM response")
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}\nResponse: {response_text}")


# --- Local Testing Block ---
if __name__ == "__main__":
    test_steps = [
        "Create a requirements.txt with FastAPI and uvicorn",
        "Create a basic Python file called app.py with a hello world endpoint",
        "Create a README.md with project description",
    ]
    
    print("Testing Code Generator...")
    for step in test_steps:
        try:
            plan = generate_execution_plan(step)
            print(f"\n✓ Generated plan for: {step}")
            print(f"  Keys: {list(plan.keys())}")
        except Exception as e:
            print(f"\n✗ Failed for: {step}")
            print(f"  Error: {e}")
